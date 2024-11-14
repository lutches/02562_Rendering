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
        code: await (await fetch("./p2.wgsl")).text()
    });
    const uniformBuffer = device.createBuffer({
        size: 16,
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

    var sphereMenu = document.getElementById("sphereMenu");
    sphereMenu.addEventListener("change", function (ev) {
        uniforms[2] = sphereMenu.value;
        animate();
    });

    var materialMenu = document.getElementById("materialMenu");
    materialMenu.addEventListener("change", function (ev) {
        uniforms[3] = materialMenu.value;
        animate();
    });


    // Create a render pass in a command buffer and submit it
    const aspect = canvas.width / canvas.height;
    var cam_const = 1.0;
    const sphereMaterial = 4.0;
    const material = 1;
    var uniforms = new Float32Array([aspect, cam_const, sphereMaterial, material]);
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
        } else if (event.key === "ArrowDown") {
            cam_const /= 1.5;
        }
        uniforms[1] = cam_const;
        requestAnimationFrame(animate);
    });

    addEventListener("wheel", function (ev) {
        ev.preventDefault();
        let zoom = ev.deltaY > 0 ? 0.95 : 1.05;
        cam_const *= zoom;
        uniforms[1] = cam_const;
        animate()
    });

    addEventListener("resize", () => {
    });

    function updateAspectRatio() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        const aspectRatio = canvas.width / canvas.height;
        const uniformData_f = new Float32Array([aspectRatio]);
    }


    function animate() {
        device.queue.writeBuffer(uniformBuffer, 0, uniforms);
        render(device, context, pipeline, bindGroup);
    }

    animate();

    function render(device, context, pipeline, bindGroup) {
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


