
window.onload = function() {main();}

function compute_jitters(buffer, divisions, pixelsize){
  // we divide each pixel into `divisions`^2 sub-pixels
  const subpixels = divisions * divisions;
  const step = 1 / divisions;
  for(var i = 0; i < subpixels; i ++){
    // generate x and y between [0, 1]
    // these are the centers of the subpixels
    // for example, if divisions = 2
    // x will be [0.25, 0.75]
    const x = step * ((i % divisions) + 0.5);
    const y = step * (Math.floor(i / divisions) + 0.5);
    //now generate some random x,y additions in range [-step/2, step/2]
    const x_offset = step * (Math.random() - 0.5);
    const y_offset = step * (Math.random() - 0.5);
    // recenter subpixels around (0, 0)
    // stored in a vec4f for byte-aligning with WGSL
    buffer[i * 4] = (x + x_offset - 0.5) * pixelsize;
    buffer[i * 4 + 1] = (y + y_offset - 0.5) * pixelsize;
    buffer[i * 4 + 2] = 0;
    buffer[i * 4 + 3] = 0;
  }
}

async function main() {

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
  ctx.configure({device: device, format: canvasFmt});
  
  
  const wgsl = device.createShaderModule({
    code: await (await fetch("./p2.wgsl")).text()
  });
  
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: wgsl,
      entryPoint: "main_vs",
    },
    fragment: {
      module: wgsl,
      entryPoint: "main_fs",
      targets: [{format: canvasFmt}]
    },
    primitive: {
      topology: "triangle-strip",
    }
  });

  render = (dev, context, pipe, bg) => {
    const encoder = dev.createCommandEncoder();

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        storeOp: "store",
      }]
    });
    pass.setBindGroup(0, bg);

    pass.setPipeline(pipe);
    pass.draw(4);
    pass.end();
    dev.queue.submit([encoder.finish()]); 
  };
  
   



  var obj_idx = 0;
  var filename = "../objects/CornellBoxWithBlocks.obj";


  var use_texture = 1;
  const aspect = canvas.width / canvas.height;
  var cam_const = 1.0;
  var gloss_shader = 5;
  var matte_shader = 1;
  const numDivisions = 3;
  var uniforms_f = new Float32Array([aspect, cam_const]);
  var uniforms_int = new Int32Array([gloss_shader, matte_shader, use_texture, numDivisions, obj_idx]);
  
  

  const uniformBuffer_f = device.createBuffer({ 
    size: uniforms_f.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 
  });  
  const uniformBuffer_int = device.createBuffer({ 
    size: uniforms_int.byteLength, // number of bytes 
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 
  }); 

  let jitter = new Float32Array(numDivisions * numDivisions * 2 * 2); // *2 to fit in vec4f for alignment
  const jitterBuffer = device.createBuffer({
    size: jitter.byteLength, 
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


  const updateSubpixels = () => {
    uniforms_int[3] = numDivisions;
    compute_jitters(jitter, numDivisions, 1/canvas.height);
    device.queue.writeBuffer(jitterBuffer, 0, jitter);
    device.queue.writeBuffer(uniformBuffer_int, 0, uniforms_int);
  };
  updateSubpixels();





  const drawingInfo = await readOBJFile(filename, 1, true); // filename, scale, ccw vertices

  const bspTreeBuffers = build_bsp_tree(drawingInfo, device, {})



  const lightIndicesBuffer = device.createBuffer({
    size: drawingInfo.light_indices.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });

  device.queue.writeBuffer(lightIndicesBuffer, 0, drawingInfo.light_indices);

  
  // flatten into diffuse, then emitted
  const mats = drawingInfo.materials.map((m) => [m.color, m.emission]).flat().map((color) => [color.r, color.g, color.b, color.a]).flat();


  const materialsArray = new Float32Array(mats);

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
    { binding: 2, resource: { buffer: jitterBuffer } },
    { binding: 3, resource: { buffer: bspTreeBuffers.aabb }},
    // storage buffers (max 8!)
    { binding: 4, resource: { buffer: bspTreeBuffers.attribs }},
    { binding: 6, resource: { buffer: materialsBuffer } },
    { binding: 7, resource: { buffer: bspTreeBuffers.indices } },
    { binding: 8, resource: { buffer: bspTreeBuffers.treeIds }},
    { binding: 9, resource: { buffer: bspTreeBuffers.bspTree }},
    { binding: 10, resource: { buffer: bspTreeBuffers.bspPlanes }},
    { binding: 11, resource: { buffer: lightIndicesBuffer}},

  ], 
  });

  function animate(){
    render(device, ctx, pipeline, bindGroup);
  }


  requestAnimationFrame(animate);

}

