"use strict";

window.onload = function () { main(); }



async function main() {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const canvas = document.getElementById("webgpu-canvas");
    const context = canvas.getContext("gpupresent") || canvas.getContext("webgpu");
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device: device, format: canvasFormat, });


    const wgsl = device.createShaderModule({
        code: await (await fetch("./week2.wgsl")).text()
    });
    const uniformBuffer = device.createBuffer({
        size: 12,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });



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


    // Create a render pass in a command buffer and submit it
    const aspect = canvas.width / canvas.height;
    var cam_const = 1.0;
    var uniforms = new Float32Array([aspect, cam_const, 1.0]);
    device.queue.writeBuffer(uniformBuffer, 0, uniforms);




    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer }
        }],
    });
    addEventListener("keydown", (event) => {
        if (event.key === "ArrowUp") {
            cam_const *= 1.5;
            requestAnimationFrame(animate);
        } else if (event.key === "ArrowDown") {
            cam_const /= 1.5;
            requestAnimationFrame(animate);
        }
        console.log(cam_const);

    });

    function animate() {
        uniforms[1] = cam_const;
        device.queue.writeBuffer(uniformBuffer, 0, uniforms);
        render(device, context, pipeline, bindGroup);
    }

    animate();

    function render(device, context, pipeline, bindGroup) 
    {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass(
            {
            colorAttachments: 
                [{
                view: context.getCurrentTexture().createView(),
                loadOp: "clear",
                storeOp: "store",
                }]
            });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(4);
        pass.end();
        device.queue.submit([encoder.finish()])
    }
}


