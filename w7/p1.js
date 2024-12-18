window.onload = function () {
  main();
};

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

  const canvas = document.querySelector("canvas");
  const ctx = canvas.getContext("webgpu");
  const canvasFmt = navigator.gpu.getPreferredCanvasFormat();

  ctx.configure({
    device: device,
    format: canvasFmt,
  });

  const wgslCode = await (await fetch("./p1.wgsl")).text();
  const shaderModule = device.createShaderModule({
    code: wgslCode,
  });

  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: shaderModule,
      entryPoint: "main_vs",
    },
    fragment: {
      module: shaderModule,
      entryPoint: "main_fs",
      targets: [
        { format: canvasFmt },
        { format: "rgba32float" }, // Output to texture for progressive rendering
      ],
    },
    primitive: {
      topology: "triangle-strip",
    },
  });

  const textures = {
    width: canvas.width,
    height: canvas.height,
    renderSrc: device.createTexture({
      size: [canvas.width, canvas.height],
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      format: "rgba32float",
    }),
    renderDst: device.createTexture({
      size: [canvas.width, canvas.height],
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      format: "rgba32float",
    }),
  };

  const render = (dev, context, pipe, texs, bg) => {
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
        },
      ],
    });

    pass.setBindGroup(0, bg);
    pass.setPipeline(pipe);
    pass.draw(4);
    pass.end();

    encoder.copyTextureToTexture(
      { texture: texs.renderSrc },
      { texture: texs.renderDst },
      [texs.width, texs.height]
    );

    dev.queue.submit([encoder.finish()]);
  };

  const filename = "../objects/CornellBoxWithBlocks.obj";

  const aspect = canvas.width / canvas.height;
  const gamma = 1.5;
  const uniforms_f = new Float32Array([aspect, gamma]);
  const uniforms_int = new Int32Array([canvas.width, canvas.height, 0]);

  const uniformBuffer_f = device.createBuffer({
    size: uniforms_f.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const uniformBuffer_int = device.createBuffer({
    size: uniforms_int.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(uniformBuffer_f, 0, uniforms_f);
  device.queue.writeBuffer(uniformBuffer_int, 0, uniforms_int);

  console.log("Load OBJ File " + filename);

  const drawingInfo = await readOBJFile(filename, 1, true); // filename, scale, CCW vertices
  console.log("Start building tree");

  const bspTreeBuffers = build_bsp_tree(drawingInfo, device, {});
  console.log("Done building tree");



  const lightIndicesBuffer = device.createBuffer({
    size: drawingInfo.light_indices.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(lightIndicesBuffer, 0, drawingInfo.light_indices);

  const materialsArray = new Float32Array(
    drawingInfo.materials
      .map((m) => [m.color, m.emission])
      .flat()
      .map((color) => [color.r, color.g, color.b, color.a])
      .flat()
  );

  const materialsBuffer = device.createBuffer({
    size: materialsArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(materialsBuffer, 0, materialsArray);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer_f } },
      { binding: 1, resource: { buffer: uniformBuffer_int } },
      { binding: 3, resource: { buffer: bspTreeBuffers.aabb } },
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
    uniforms_int[2]++;
    device.queue.writeBuffer(uniformBuffer_int, 0, uniforms_int);
    render(device, ctx, pipeline, textures, bindGroup);
  }

  const frameCounter = document.getElementById("framecount");
  let running = false;

  document.getElementById("run").onclick = () => {
    running = !running;
  };

  setInterval(() => {
    frameCounter.innerText = "Frame: " + uniforms_int[2];
    if (running) {
      requestAnimationFrame(animate);
    }
  }, 1);
  animate();
}