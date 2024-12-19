window.onload = function () {
  main();
};

async function loadTexture(device, filename) {
  if (typeof filename !== 'string' || !filename.trim()) {
    throw new Error("Invalid filename. Please provide a valid texture file path.");
  }

  try {
    // Fetch the image file
    const response = await fetch(filename);
    if (!response.ok) {
      throw new Error(`Failed to fetch texture: ${response.statusText}`);
    }

    const blob = await response.blob();

    // Create image bitmap
    const img = await createImageBitmap(blob, { colorSpaceConversion: 'none' });

    // Ensure texture size is valid
    if (!img.width || !img.height) {
      throw new Error("Invalid image dimensions for texture creation.");
    }

    // Create texture
    const texture = device.createTexture({
      size: [img.width, img.height, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // Copy image bitmap to the texture
    device.queue.copyExternalImageToTexture(
      { source: img, flipY: true },
      { texture },
      { width: img.width, height: img.height },
    );

    return texture;
  } catch (error) {
    console.error("Error loading texture:", error);
    throw error;
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

  // Load WGSL shader code
  const wgsl = device.createShaderModule({
    code: await (await fetch("./p3.wgsl")).text()
  });

  const bg_texture = await loadTexture(device, "../backgrounds/luxo_pxr_campus.jpg");

  // Setup canvas and configure context
  const canvas = document.querySelector("canvas");
  const ctx = canvas.getContext("webgpu");
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({
    device: device,
    format: canvasFormat
  });

  // Create pipeline
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
        { format: canvasFormat },
        { format: "rgba32float" } // Second render target for progressive rendering
      ]
    },
    primitive: {
      topology: "triangle-strip",
    }
  });

  // Setup textures
  const textures = {
    width: canvas.width,
    height: canvas.height,
    renderSrc: device.createTexture({
      size: [canvas.width, canvas.height],
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      format: 'rgba32float',
    }),
    renderDst: device.createTexture({
      size: [canvas.width, canvas.height],
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      format: 'rgba32float',
    })
  };

  // Render function
  function render(dev, context, pipe, texs, bindGroup) {
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
    pass.setBindGroup(0, bindGroup);
    pass.setPipeline(pipe);
    pass.draw(4);
    pass.end();

    encoder.copyTextureToTexture(
      { texture: texs.renderSrc },
      { texture: texs.renderDst },
      [texs.width, texs.height]
    );

    dev.queue.submit([encoder.finish()]);
  }

  // Load scene geometry
  const filename = "../objects/bunny.obj";
  const drawingInfo = await readOBJFile(filename, 30, true);

  // Prepare uniform data
  const aspect = canvas.width / canvas.height;
  const gamma = 1.5;
  var shader = 1;
  const uniforms_f = new Float32Array([aspect, gamma]);

  let frame_num = 0;
  const uniforms_int = new Int32Array([canvas.width, canvas.height, frame_num, shader]);

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

  // Build BSP tree and related buffers
  const bspTreeBuffers = build_bsp_tree(drawingInfo, device, {});

  // Create light indices buffer
  const lightIndicesBuffer = device.createBuffer({
    size: drawingInfo.light_indices.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(lightIndicesBuffer, 0, drawingInfo.light_indices);

  // Flatten and upload material data
  const mats = drawingInfo.materials
    .map(m => [m.color, m.emission])
    .flat()
    .map(color => [color.r, color.g, color.b, color.a])
    .flat();
  const materialsArray = new Float32Array(mats);

  const materialsBuffer = device.createBuffer({
    size: materialsArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(materialsBuffer, 0, materialsArray);

  

  // Create bind group
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
      { binding: 12, resource: textures.renderDst.createView() },
      { binding: 13, resource: bg_texture.createView() },
    ],
  });

  if (document.querySelector('input[name="glossy"]')) {
    document.querySelectorAll('input[name="glossy"]').forEach((elem) => {
      elem.addEventListener("change", function (event) {
        shader = event.target.value;
        frame_num = 0;
        requestAnimationFrame(animate);
      });
    });
  }

  // Animation loop
  function animate() {
    uniforms_int[2] = frame_num;
    uniforms_int[3] = shader;
    frame_num++;
    device.queue.writeBuffer(uniformBuffer_int, 0, uniforms_int);
    render(device, ctx, pipeline, textures, bindGroup);
  }

  const frameCounter = document.getElementById("framecount");
  let running = false;

  document.getElementById("run").onclick = () => {
    running = !running;
  };

  // Update frames at a set interval
  setInterval(() => {
    frameCounter.innerText = "Frame: " + frame_num;
    if (running && frame_num < 4000) {
      requestAnimationFrame(animate);
    }
  }, 1);
  requestAnimationFrame(animate);
}
