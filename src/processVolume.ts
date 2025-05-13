export class DiffusionSim {
    public device: GPUDevice;
    private dimensions: { x: number, y: number, z: number };
    private inputBuffer: GPUBuffer;
    private outputBuffer: GPUBuffer;
    private stagingBuffer: GPUBuffer;
    private dimensionsBuffer: GPUBuffer;
    private computePipeline: GPUComputePipeline;
    private bindGroup: GPUBindGroup;
    public readPromise: Promise<Float32Array | null> | null = null;
    private diffusionConstantBuffer: GPUBuffer;

    constructor(device: GPUDevice, dimensions: { x: number, y: number, z: number }, inputBuffer: GPUBuffer, outputBuffer: GPUBuffer, stagingBuffer: GPUBuffer, dimensionsBuffer: GPUBuffer, computePipeline: GPUComputePipeline, bindGroup: GPUBindGroup, diffusionConstantBuffer: GPUBuffer) {
        this.device = device;
        this.dimensions = dimensions;
        this.inputBuffer = inputBuffer;
        this.outputBuffer = outputBuffer;
        this.stagingBuffer = stagingBuffer;
        this.dimensionsBuffer = dimensionsBuffer;
        this.computePipeline = computePipeline;
        this.bindGroup = bindGroup;
        this.diffusionConstantBuffer = diffusionConstantBuffer;
    }

    setDiffusionConstant(value: number) {
        this.device.queue.writeBuffer(this.diffusionConstantBuffer, 0, new Float32Array([value]));
    }

    static async create(dimensions: { x: number, y: number, z: number }, volumeData: Float32Array, diffusionConstant: number) {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('No WebGPU adapter found');
        }

        // Request a higher maxComputeInvocationsPerWorkgroup limit
        const device = await adapter.requestDevice({
            requiredLimits: {
                maxComputeInvocationsPerWorkgroup: 1024 // This should be enough for our 8x8x8 workgroup
            }
        });

        // Create input and output buffers
        const inputBuffer = device.createBuffer({
            size: volumeData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const outputBuffer = device.createBuffer({
            size: volumeData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Create staging buffer for reading back results
        const stagingBuffer = device.createBuffer({
            size: volumeData.byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const dimensionsBuffer = device.createBuffer({
            size: 3 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const diffusionConstantBuffer = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        device.queue.writeBuffer(dimensionsBuffer, 0, new Uint32Array([dimensions.x, dimensions.y, dimensions.z]));

        // Write input data to buffer
        device.queue.writeBuffer(inputBuffer, 0, volumeData);

        device.queue.writeBuffer(diffusionConstantBuffer, 0, new Float32Array([diffusionConstant]));

        // Create compute pipeline
        const computePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: computeShader,
                }),
                entryPoint: 'main',
            },
        });

        // Create bind group
        const bindGroup = device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: inputBuffer } },
                { binding: 1, resource: { buffer: outputBuffer } },
                { binding: 2, resource: { buffer: dimensionsBuffer } },
                { binding: 3, resource: { buffer: diffusionConstantBuffer } },
            ],
        });

        return new DiffusionSim(device, dimensions, inputBuffer, outputBuffer, stagingBuffer, dimensionsBuffer, computePipeline, bindGroup, diffusionConstantBuffer);
    }

    async process() {
        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.bindGroup);

        const workgroupSize = 8;
        const dimX = Math.ceil(this.dimensions.x / workgroupSize);
        const dimY = Math.ceil(this.dimensions.y / workgroupSize);
        const dimZ = Math.ceil(this.dimensions.z / workgroupSize);
        computePass.dispatchWorkgroups(dimX, dimY, dimZ);
        computePass.end();

        commandEncoder.copyBufferToBuffer(this.outputBuffer, 0, this.inputBuffer, 0, this.inputBuffer.size);

        this.device.queue.submit([commandEncoder.finish()]);
    }

    async readResults(): Promise<Float32Array | null> {
        // If there's already a read in progress, wait for it to complete
        if (this.readPromise) {
            return this.readPromise;
        }

        // Create a new read operation
        this.readPromise = (async () => {
            // First copy the output to staging buffer
            const commandEncoder = this.device.createCommandEncoder();
            commandEncoder.copyBufferToBuffer(this.outputBuffer, 0, this.stagingBuffer, 0, this.outputBuffer.size);
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            // Now we can safely map and read
            await this.stagingBuffer.mapAsync(GPUMapMode.READ);
            const result = new Float32Array(this.stagingBuffer.getMappedRange());
            const processedData = new Float32Array(result);
            this.stagingBuffer.unmap();
            this.readPromise = null;
            return processedData;
        })();

        return this.readPromise;
    }

    async cleanup() {
        this.inputBuffer.destroy();
        this.outputBuffer.destroy();
        this.stagingBuffer.destroy();
        this.dimensionsBuffer.destroy();
    }
}

const computeShader = /* wgsl */`
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> dimensions: vec3<u32>;
@group(0) @binding(3) var<storage, read> diffusionConstant: f32;

fn xyz_to_index(x: u32, y: u32, z: u32) -> u32 {
    return x + y * dimensions.x + z * dimensions.x * dimensions.y;
}

const DeltaTime = 1.0/60.0;
const DeltaSpace = 1.0;

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // get position and index
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    let idx = xyz_to_index(x, y, z);

    // get values of neighbors and center
    let x_prev_val = select(input[xyz_to_index(x - 1u, y, z)], 0, x - 1u == 0u);
    let x_next_val = select(input[xyz_to_index(x + 1u, y, z)], 0, x + 1u == dimensions.x - 1u);
    let y_prev_val = select(input[xyz_to_index(x, y - 1u, z)], 0, y - 1u == 0u);
    let y_next_val = select(input[xyz_to_index(x, y + 1u, z)], 0, y + 1u == dimensions.y - 1u);
    let z_prev_val = select(input[xyz_to_index(x, y, z - 1u)], 0, z - 1u == 0u);
    let z_next_val = select(input[xyz_to_index(x, y, z + 1u)], 0, z + 1u == dimensions.z - 1u);
    let center_val = input[idx];

    // calculate diffusion
    let diffusion = diffusionConstant * (DeltaTime / (DeltaSpace * DeltaSpace)) * 
        (x_prev_val + x_next_val + y_prev_val + y_next_val +
        z_prev_val + z_next_val - 6.0 * center_val);
    
    // Add point source in center and apply diffusion
    // let isCenter = x == dimensions.x / 2u && y == dimensions.y / 2u && z == dimensions.z / 2u;
    let sourceValue = select(0.0, 1.0, x == dimensions.x / 2u);
    output[idx] = center_val + diffusion + sourceValue;
}
`;