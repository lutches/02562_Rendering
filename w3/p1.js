"use strict";

window.onload = function () { main(); }

var uniforms_f = new Float32Array([1.0, 0.0, 0.0, 0.0]);

async function main() {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const canvas = document.getElementById("webgpu-canvas");
    const context = canvas.getContext("gpupresent") || canvas.getContext("webgpu");
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device: device, format: canvasFormat, });


    const wgsl = device.createShaderModule({
        code: await (await fetch("./p1.wgsl")).text()
    });
    const uniformBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    var addressMenu = document.getElementById("addressmode");
    var filterMenu = document.getElementById("filtermode");
    const use_repeat = addressMenu.selectedIndex;
    const use_linear = filterMenu.selectedIndex;
    var uniforms_ui = new Uint32Array([use_repeat, use_linear]);

    const uniformBuffer_ui = device.createBuffer({
        size: uniforms_ui.byteLength, // number of bytes
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer_ui, 0, uniforms_ui);

    const uniformBuffer_f = device.createBuffer({
        size: 4, // 4 bytes for a single f32 (float)
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    addressMenu.addEventListener("change", () => {
        uniforms_ui[0] = addressMenu.selectedIndex;
        device.queue.writeBuffer(uniformBuffer_ui, 0, uniforms_ui);
        requestAnimationFrame(animate);
    });

    filterMenu.addEventListener("change", () => {
        uniforms_ui[1] = filterMenu.selectedIndex;
        device.queue.writeBuffer(uniformBuffer_ui, 0, uniforms_ui);
        requestAnimationFrame(animate);
    });

    window.addEventListener("resize", () => {
        updateAspectRatio();
        requestAnimationFrame(animate); // Re-render with new aspect ratio
    });

    function updateAspectRatio() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight; 
        
        const aspectRatio = canvas.width / canvas.height;
        const uniformData_f = new Float32Array([aspectRatio]);
        device.queue.writeBuffer(uniformBuffer_f, 0, uniformData_f);
    }
    
    // Call updateAspectRatio initially and whenever the canvas size changes
    updateAspectRatio();


    const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex:
        {
            module: wgsl,
            entryPoint: "main_vs",
        },
        fragment: {
            module: wgsl,
            entryPoint: "main_fs",
            targets: [{ format: canvasFormat }]
        },
        primitive: {
            topology: "triangle-strip",
        },
    });

    
    

    const texture = await load_texture(device, "../textures/grass.jpg");

    console.log(pipeline.getBindGroupLayout(0));
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer_f } },
            { binding: 1, resource: { buffer: uniformBuffer_ui } },
            { binding: 2, resource: texture.createView() },
        ],
    });

    function render(device, context, pipeline, bindGroup) {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                loadOp: "clear",
                storeOp: "store",
            }]
        });

        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(4);
        pass.end();
        device.queue.submit([encoder.finish()]);
    }

    function animate() {
        render(device, context, pipeline, bindGroup);
    }
    animate();
    
    async function load_texture(device, filename) {
        const response = await fetch(filename);
        const blob = await response.blob();
        const img = await createImageBitmap(blob, { colorSpaceConversion: 'none' });

        const texture = device.createTexture({
            size: [img.width, img.height, 1],
            format: "rgba8unorm",
            usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
        });

        device.queue.copyExternalImageToTexture(
            { source: img, flipY: true },
            { texture: texture },
            { width: img.width, height: img.height },
        );
        return texture;
    }


}


