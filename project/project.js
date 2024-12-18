window.onload = () => main();

async function main() {
  // Ensure WebGPU support
  if (!navigator.gpu) throw new Error("WebGPU is not supported in this browser.");

  // Initialize device and context
  const { device, context, canvasFormat } = await initWebGPU();
  const shaderModule = await createShaderModule(device, "./project.wgsl");
  const pipeline = createPipeline(device, shaderModule, canvasFormat);

  // Create textures for progressive rendering
  const textures = createTextures(device, context.canvas);

  // Load OBJ data and build BSP tree
  const filename = "../objects/CornellBoxWithBlocks.obj";
  const drawingInfo = await readOBJFile(filename, 1, true);
  const bspTreeBuffers = build_bsp_tree(drawingInfo, device, {});

  // Create buffers and bind group
  const { uniformBuffer_f, uniformBuffer_int, materialsBuffer, lightIndicesBuffer } =
    createBuffers(device, context.canvas, drawingInfo);

  const bindGroup = createBindGroup(
    device, pipeline, textures, bspTreeBuffers,
    uniformBuffer_f, uniformBuffer_int, materialsBuffer, lightIndicesBuffer
  );

  // Animation and rendering loop
  startAnimationLoop(device, context, pipeline, textures, bindGroup, uniformBuffer_int);
}

async function initWebGPU() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No suitable GPUAdapter found.");

  const device = await adapter.requestDevice();
  const canvas = document.querySelector("canvas");
  const context = canvas.getContext("webgpu");
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format: canvasFormat });

  return { device, context, canvasFormat };
}

async function createShaderModule(device, url) {
  const response = await fetch(url);
  const code = await response.text();
  return device.createShaderModule({ code });
}

function createPipeline(device, shaderModule, canvasFormat) {
  return device.createRenderPipeline({
    layout: "auto",
    vertex: { module: shaderModule, entryPoint: "main_vs" },
    fragment: {
      module: shaderModule,
      entryPoint: "main_fs",
      targets: [
        { format: canvasFormat },
        { format: "rgba32float" }, // For progressive rendering
      ],
    },
    primitive: { topology: "triangle-strip" },
  });
}

function createTextures(device, canvas) {
  return {
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
}

function createBuffers(device, canvas, drawingInfo) {
  const aspectRatio = canvas.width / canvas.height;
  const gamma = 1.5;

  const uniforms_f = new Float32Array([aspectRatio, gamma]);
  const uniforms_int = new Int32Array([canvas.width, canvas.height, 0]); // frame_num = 0

  const uniformBuffer_f = device.createBuffer({
    size: uniforms_f.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer_f, 0, uniforms_f);

  const uniformBuffer_int = device.createBuffer({
    size: uniforms_int.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer_int, 0, uniforms_int);

  const materialsArray = new Float32Array(
    drawingInfo.materials
      .flatMap((m) => [m.color, m.emission])
      .flatMap((c) => [c.r, c.g, c.b, c.a])
  );

  const materialsBuffer = device.createBuffer({
    size: materialsArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(materialsBuffer, 0, materialsArray);

  const lightIndicesBuffer = device.createBuffer({
    size: drawingInfo.light_indices.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(lightIndicesBuffer, 0, drawingInfo.light_indices);

  return { uniformBuffer_f, uniformBuffer_int, materialsBuffer, lightIndicesBuffer };
}

function createBindGroup(
  device, pipeline, textures, bspTreeBuffers,
  uniformBuffer_f, uniformBuffer_int, materialsBuffer, lightIndicesBuffer
) {
  return device.createBindGroup({
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
}

function renderFrame(device, context, pipeline, textures, bindGroup) {
  const encoder = device.createCommandEncoder();
  const renderPass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        storeOp: "store",
      },
      {
        view: textures.renderSrc.createView(),
        loadOp: "load",
        storeOp: "store",
      },
    ],
  });

  renderPass.setBindGroup(0, bindGroup);
  renderPass.setPipeline(pipeline);
  renderPass.draw(4);
  renderPass.end();

  encoder.copyTextureToTexture(
    { texture: textures.renderSrc },
    { texture: textures.renderDst },
    [textures.width, textures.height]
  );

  device.queue.submit([encoder.finish()]);
}

function startAnimationLoop(device, context, pipeline, textures, bindGroup, uniformBuffer_int) {
  let frameNum = 0;
  let running = true;
  const frameCounter = document.getElementById("framecount");
  const button = document.getElementById("run");

  button.onclick = () => running = !running;

  const animate = () => {
    const uniforms_int = new Int32Array([textures.width, textures.height, frameNum++]);
    device.queue.writeBuffer(uniformBuffer_int, 0, uniforms_int);
    renderFrame(device, context, pipeline, textures, bindGroup);
  };

  setInterval(() => {
    frameCounter.innerText = `Frame: ${frameNum}`;
    if (running) requestAnimationFrame(animate);
  }, 1);

  animate();
}
