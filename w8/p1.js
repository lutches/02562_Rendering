
window.onload = function () { main(); }

async function main() {
  console.log("Begin main function");
  if (!navigator.gpu) {
    throw new Error("WebGPU not supported on this browser.");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.");
  }

  const device = await adapter.requestDevice();

  const wgsl = device.createShaderModule({
    code: await (await fetch("./p1.wgsl")).text()
  });

  const canvas = document.querySelector("canvas");
  const ctx = canvas.getContext("webgpu");
  const canvasFmt = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device: device, format: canvasFmt });




  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: wgsl,
      entryPoint: "main_vs",
    },
    fragment: {
      module: wgsl,
      entryPoint: "main_fs",
      targets: [
        { format: canvasFmt },
        { format: "rgba32float" } // output to texture for progressive rendering
      ]
    },
    primitive: {
      topology: "triangle-strip",
    }
  });

  let textures = new Object();
  textures.width = canvas.width;
  textures.height = canvas.height;
  textures.renderSrc = device.createTexture({
    size: [canvas.width, canvas.height],
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    format: 'rgba32float',
  });

  textures.renderDst = device.createTexture({
    size: [canvas.width, canvas.height],
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    format: 'rgba32float',
  })

  render = (dev, context, pipe, texs, bg) => {
    const encoder = dev.createCommandEncoder();

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
        },
        {
          view: texs.renderSrc.createView(),
          loadOp: "load",
          storeOp: "store",
        }
      ]
    });
    pass.setBindGroup(0, bg);

    pass.setPipeline(pipe);
    pass.draw(4);
    pass.end();

    encoder.copyTextureToTexture(
      { texture: texs.renderSrc },
      { texture: texs.renderDst },
      [textures.width, textures.height]
    );

    dev.queue.submit([encoder.finish()]);
  };


  var filename = "../objects/CornellBox.obj";


  const aspect = canvas.width / canvas.height;
  const gamma = 1.5;
  var uniforms_f = new Float32Array([aspect, gamma]);
  var frame_num = 0;
  var uniforms_int = new Int32Array([canvas.width, canvas.height, frame_num]);



  const uniformBuffer_f = device.createBuffer({
    size: uniforms_f.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const uniformBuffer_int = device.createBuffer({
    size: uniforms_int.byteLength, // number of bytes 
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });



  // bspTreeBuffers has the following:
  // - attribs (positions + normals?)
  // - colors
  // - indices
  // - treeIds
  // - bspTree
  // - bspPlanes
  // - aabb (stored in uniform buffer)

  // the build_bsp_tree function creates these buffer on the device
  // all we have to do is put them in the right spots in the bindGroup layout




  device.queue.writeBuffer(uniformBuffer_f, 0, uniforms_f);
  device.queue.writeBuffer(uniformBuffer_int, 0, uniforms_int);




  console.log("Load OBJ File " + filename);

  const drawingInfo = await readOBJFile(filename, 1, true); // filename, scale, ccw vertices
  console.log("Start building tree");
  const bspTreeBuffers = build_bsp_tree(drawingInfo, device, {})
  console.log("done building tree");


  console.log("indices:");
  console.log(drawingInfo.indices);
  // To see every 4th element (material indices):
  console.log("material indices:");
  console.log(drawingInfo.indices.filter((_, i) => i % 4 === 3));
  console.log("vertices:");
  console.log(drawingInfo.attribs.filter((_, i) => i % 8 < 3));

  console.log("AABB:", root.bbox);
  const lightIndicesBuffer = device.createBuffer({
    size: drawingInfo.light_indices.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });

  device.queue.writeBuffer(lightIndicesBuffer, 0, drawingInfo.light_indices);


  // flatten into diffuse, then emitted
  const mats = drawingInfo.materials.map((m) => [m.color, m.emission]).flat().map((color) => [color.r, color.g, color.b, color.a]).flat();
  console.log("Mats");
  console.log(mats);
  const materialsArray = new Float32Array(mats);
  console.log(materialsArray);
  const materialsBuffer = device.createBuffer({
    size: materialsArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(materialsBuffer, 0, materialsArray);


  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      // uniforms 
      { binding: 0, resource: { buffer: uniformBuffer_f } },
      { binding: 1, resource: { buffer: uniformBuffer_int } },
      { binding: 3, resource: { buffer: bspTreeBuffers.aabb } },
      // storage buffers (max 8!)
      { binding: 4, resource: { buffer: bspTreeBuffers.attribs } },
      { binding: 6, resource: { buffer: materialsBuffer } },
      { binding: 7, resource: { buffer: bspTreeBuffers.indices } },
      { binding: 8, resource: { buffer: bspTreeBuffers.treeIds } },
      { binding: 9, resource: { buffer: bspTreeBuffers.bspTree } },
      { binding: 10, resource: { buffer: bspTreeBuffers.bspPlanes } },
      { binding: 11, resource: { buffer: lightIndicesBuffer } },
      { binding: 12, resource: textures.renderDst.createView() },

    ],
  });


  function animate() {
    console.log(frame_num);
    uniforms_int[2] = frame_num;
    frame_num++;
    device.queue.writeBuffer(uniformBuffer_int, 0, uniforms_int);
    render(device, ctx, pipeline, textures, bindGroup);
  }

  const frameCounter = document.getElementById("framecount");

  requestAnimationFrame(animate);

  var running = false;
  document.getElementById("run").onclick = () => {
    running = !running;
  };

  // every millisecond, request a new frame
  setInterval(() => {
    frameCounter.innerText = "Frame: " + frame_num;
    if (running) {
      requestAnimationFrame(animate);
    }

  }, 1);

}

