import { Cell } from "./cells";

export class DiffusionSim {
    public device: GPUDevice;
    private dimensions: { x: number, y: number, z: number };

    private stagingBuffer: GPUBuffer;
    private dimensionsBuffer: GPUBuffer;
    private combinedConstantBuffer: GPUBuffer;
    private deltaTimeBuffer: GPUBuffer;
    private cellsBuffer: GPUBuffer;

    private diffusionComputePipeline: GPUComputePipeline;
    private diffusionBindGroup: GPUBindGroup;

    public readPromise: Promise<Float32Array | null> | null = null;

    private concentrationBuffer: GPUBuffer;
    private concentrationOutputBuffer: GPUBuffer;
    private cellIndexBuffer: GPUBuffer;

    constructor(
        device: GPUDevice,
        dimensions: { x: number, y: number, z: number },
        concentrationBuffer: GPUBuffer,
        concentrationOutputBuffer: GPUBuffer,
        cellIndexBuffer: GPUBuffer,
        stagingBuffer: GPUBuffer,
        dimensionsBuffer: GPUBuffer,
        cellsBuffer: GPUBuffer,
        combinedConstantBuffer: GPUBuffer,
        deltaTimeBuffer: GPUBuffer,
        diffusionComputePipeline: GPUComputePipeline,
        diffusionBindGroup: GPUBindGroup) {
        this.device = device;
        this.dimensions = dimensions;

        this.concentrationBuffer = concentrationBuffer;
        this.concentrationOutputBuffer = concentrationOutputBuffer;
        this.cellIndexBuffer = cellIndexBuffer;
        this.stagingBuffer = stagingBuffer;
        this.dimensionsBuffer = dimensionsBuffer;
        this.combinedConstantBuffer = combinedConstantBuffer;
        this.deltaTimeBuffer = deltaTimeBuffer;
        this.cellsBuffer = cellsBuffer;

        this.diffusionComputePipeline = diffusionComputePipeline;
        this.diffusionBindGroup = diffusionBindGroup;
    }

    setCombinedConstant(diffusionConstant: number, deltaTime: number) {
        const deltaSpace = 1.0;
        this.device.queue.writeBuffer(this.combinedConstantBuffer, 0, new Float32Array([diffusionConstant * (deltaTime / (deltaSpace * deltaSpace))]));
    }

    setDeltaTime(value: number) {
        this.device.queue.writeBuffer(this.deltaTimeBuffer, 0, new Float32Array([value]));
    }

    static flattenCells(cells: Cell[]) {
        const cellData = new Float32Array(cells.length * 4); // 4 floats per cell: x,y,z,production
        for (let i = 0; i < cells.length; i++) {
            const cell = cells[i];
            const baseIndex = i * 4;
            cellData[baseIndex] = cell.position.x;
            cellData[baseIndex + 1] = cell.position.y;
            cellData[baseIndex + 2] = cell.position.z;
            cellData[baseIndex + 3] = cell.productionRate;
        }
        return cellData;
    }

    updateCells(cells: Cell[]) {
        const cellData = DiffusionSim.flattenCells(cells);

        // Destroy the old buffer
        this.cellsBuffer.destroy();

        // Create a new buffer with the correct size
        this.cellsBuffer = this.device.createBuffer({
            label: "Cells data buffer",
            size: Math.max(cells.length * 4 * 4, 4), // 4 floats per cell * 4 bytes per float
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Write the new cell data
        this.device.queue.writeBuffer(this.cellsBuffer, 0, cellData);

        let cellIndexData = new Int32Array(this.dimensions.x * this.dimensions.y * this.dimensions.z);
        cellIndexData.fill(-1);

        for (let i = 0; i < cells.length; i++) {
            const cell = cells[i];
            const cellPosition = cell.position;
            const cellIndex = cellPosition.x + cellPosition.y * this.dimensions.x + cellPosition.z * this.dimensions.x * this.dimensions.y;

            cellIndexData[cellIndex] = i;
        }

        // Destroy the old buffer
        this.cellIndexBuffer.destroy();

        // Create a new buffer with the correct size
        this.cellIndexBuffer = this.device.createBuffer({
            label: "Cell index buffer",
            size: cellIndexData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Write the new cell index data
        this.device.queue.writeBuffer(this.cellIndexBuffer, 0, cellIndexData);


        // Recreate the bind group with the new buffer
        this.diffusionBindGroup = this.device.createBindGroup({
            label: "Diffusion bind group",
            layout: this.diffusionComputePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.concentrationBuffer } },
                { binding: 1, resource: { buffer: this.concentrationOutputBuffer } },
                { binding: 2, resource: { buffer: this.dimensionsBuffer } },
                { binding: 3, resource: { buffer: this.combinedConstantBuffer } },
                { binding: 4, resource: { buffer: this.deltaTimeBuffer } },
                { binding: 5, resource: { buffer: this.cellIndexBuffer } },
                { binding: 6, resource: { buffer: this.cellsBuffer } },
            ],
        });
    }

    static async create(
        dimensions: { x: number, y: number, z: number },
        concentrationData: Float32Array,
        diffusionConstant: number,
        deltaTime: number,
        cells: Cell[]) {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('No WebGPU adapter found');
        }

        const requiredLimits = {
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxBufferSize: adapter.limits.maxBufferSize,
            maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup
        }
        const device = await adapter.requestDevice({
            requiredLimits: requiredLimits
        });

        // Create concentration buffers
        const concentrationBuffer = device.createBuffer({
            label: "Concentration buffer",
            size: concentrationData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        device.queue.writeBuffer(concentrationBuffer, 0, concentrationData);

        const concentrationOutputBuffer = device.createBuffer({
            label: "Concentration output buffer",
            size: concentrationData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        // Create staging buffer for reading back results
        const stagingBuffer = device.createBuffer({
            label: "Staging buffer for reading results",
            size: concentrationData.byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const dimensionsBuffer = device.createBuffer({
            label: "Volume dimensions buffer",
            size: 3 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const combinedConstantBuffer = device.createBuffer({
            label: "Combined constant buffer",
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const deltaTimeBuffer = device.createBuffer({
            label: "Delta time buffer",
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        device.queue.writeBuffer(dimensionsBuffer, 0, new Uint32Array([dimensions.x, dimensions.y, dimensions.z]));
        device.queue.writeBuffer(deltaTimeBuffer, 0, new Float32Array([deltaTime]));

        const cellsBuffer = device.createBuffer({
            label: "Cells data buffer",
            size: Math.max(cells.length * 4 * 4, 4), // Ensure minimum size of 4 bytes (1 float) 4 floats per cell (x,y,z,production) * 4 bytes per float
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Create cell index buffer
        const cellIndexBuffer = device.createBuffer({
            label: "Cell index buffer",
            size: dimensions.x * dimensions.y * dimensions.z * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Create diffusion compute pipeline
        const diffusionComputePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: diffusionShader,
                }),
                entryPoint: 'main',
            },
        });

        // Create diffusion bind group
        const diffusionBindGroup = device.createBindGroup({
            label: "Diffusion bind group",
            layout: diffusionComputePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: concentrationBuffer } },
                { binding: 1, resource: { buffer: concentrationOutputBuffer } },
                { binding: 2, resource: { buffer: dimensionsBuffer } },
                { binding: 3, resource: { buffer: combinedConstantBuffer } },
                { binding: 4, resource: { buffer: deltaTimeBuffer } },
                { binding: 5, resource: { buffer: cellIndexBuffer } },
                { binding: 6, resource: { buffer: cellsBuffer } },
            ],
        });

        const sim = new DiffusionSim(device, dimensions,
            concentrationBuffer, concentrationOutputBuffer, cellIndexBuffer,
            stagingBuffer, dimensionsBuffer, cellsBuffer, combinedConstantBuffer,
            deltaTimeBuffer, diffusionComputePipeline, diffusionBindGroup);

        sim.updateCells(cells);
        sim.setCombinedConstant(diffusionConstant, deltaTime);
        return sim;
    }

    async process() {
        const diffusionCommandEncoder = this.device.createCommandEncoder();
        diffusionCommandEncoder.clearBuffer(this.concentrationOutputBuffer);

        const diffusionComputePass = diffusionCommandEncoder.beginComputePass();
        diffusionComputePass.setPipeline(this.diffusionComputePipeline);
        diffusionComputePass.setBindGroup(0, this.diffusionBindGroup);

        const workgroupSize = 8;
        const dimX = Math.ceil(this.dimensions.x / workgroupSize);
        const dimY = Math.ceil(this.dimensions.y / workgroupSize);
        const dimZ = Math.ceil(this.dimensions.z / workgroupSize);
        diffusionComputePass.dispatchWorkgroups(dimX, dimY, dimZ);
        diffusionComputePass.end();

        diffusionCommandEncoder.copyBufferToBuffer(
            this.concentrationOutputBuffer, 0,
            this.concentrationBuffer, 0,
            this.concentrationBuffer.size
        );

        this.device.queue.submit([diffusionCommandEncoder.finish()]);
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
            commandEncoder.copyBufferToBuffer(
                this.concentrationOutputBuffer, 0,
                this.stagingBuffer, 0,
                this.concentrationOutputBuffer.size
            );
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
        this.concentrationBuffer.destroy();
        this.concentrationOutputBuffer.destroy();
        this.cellIndexBuffer.destroy();
        this.stagingBuffer.destroy();
        this.dimensionsBuffer.destroy();
        this.combinedConstantBuffer.destroy();
        this.cellsBuffer.destroy();
        this.deltaTimeBuffer.destroy();
    }
}

const diffusionShader = /* wgsl */`
@group(0) @binding(0) var<storage, read> input_concentration: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_concentration: array<f32>;
@group(0) @binding(2) var<uniform> dimensions: vec3<u32>;
@group(0) @binding(3) var<storage, read> combined_constant: f32;
@group(0) @binding(4) var<uniform> delta_time: f32;
@group(0) @binding(5) var<storage, read> cell_indices: array<i32>;
@group(0) @binding(6) var<storage, read> cells: array<f32>;

fn xyz_to_index(x: u32, y: u32, z: u32) -> i32 {
    return i32(x + y * dimensions.x + z * dimensions.x * dimensions.y);
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // get position and index
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;
    let idx = xyz_to_index(x, y, z);

    // get neighbor indices, if out of bounds, return -1
    let x_prev_idx = select(xyz_to_index(x - 1u, y, z), -1, x == 0u);
    let x_next_idx = select(xyz_to_index(x + 1u, y, z), -1, x + 1u == dimensions.x);
    let y_prev_idx = select(xyz_to_index(x, y - 1u, z), -1, y == 0u);
    let y_next_idx = select(xyz_to_index(x, y + 1u, z), -1, y + 1u == dimensions.y);
    let z_prev_idx = select(xyz_to_index(x, y, z - 1u), -1, z == 0u);
    let z_next_idx = select(xyz_to_index(x, y, z + 1u), -1, z + 1u == dimensions.z);

    // get cell index
    let this_position_cell_idx = cell_indices[idx];
    // is this a cell?
    if (this_position_cell_idx != -1) {
        var num_open = 6;
        // if in bounds voxel that has a cell, decrement num_open
        if (x_prev_idx != -1 && cell_indices[x_prev_idx] != -1) {
            num_open -= 1;
        }
        if (x_next_idx != -1 && cell_indices[x_next_idx] != -1) {
            num_open -= 1;
        }
        if (y_prev_idx != -1 && cell_indices[y_prev_idx] != -1) {
            num_open -= 1;
        }
        if (y_next_idx != -1 && cell_indices[y_next_idx] != -1) {
            num_open -= 1;
        }
        if (z_prev_idx != -1 && cell_indices[z_prev_idx] != -1) {
            num_open -= 1;
        }
        if (z_next_idx != -1 && cell_indices[z_next_idx] != -1) {
            num_open -= 1;
        }

        // get cell production rate
        let cell_production_rate = cells[this_position_cell_idx * 4 + 3];
        let cell_production = cell_production_rate * delta_time;
        let cell_production_per_open = cell_production / f32(num_open);

        // add production
        // is this an inbounds voxel without a cell?
        if (x_prev_idx != -1 && cell_indices[x_prev_idx] == -1) {
            output_concentration[x_prev_idx] += cell_production_per_open;
        }
        if (x_next_idx != -1 && cell_indices[x_next_idx] == -1) {
            output_concentration[x_next_idx] += cell_production_per_open;
        }
        if (y_prev_idx != -1 && cell_indices[y_prev_idx] == -1) {
            output_concentration[y_prev_idx] += cell_production_per_open;
        }
        if (y_next_idx != -1 && cell_indices[y_next_idx] == -1) {
            output_concentration[y_next_idx] += cell_production_per_open;
        }
        if (z_prev_idx != -1 && cell_indices[z_prev_idx] == -1) {
            output_concentration[z_prev_idx] += cell_production_per_open;
        }
        if (z_next_idx != -1 && cell_indices[z_next_idx] == -1) {
            output_concentration[z_next_idx] += cell_production_per_open;
        }
    } else {
        var sum_open = 0.0;
        var num_open = 0;

        // if voxel is out of bounds, add 1 to num_open
        // otherwise, if voxel doesn't have a cell, add its concentration to the sum and add 1 to num_open
        if (x_prev_idx == -1) {
            num_open += 1;
        } else if (cell_indices[x_prev_idx] == -1) {
            sum_open += input_concentration[x_prev_idx];
            num_open += 1;
        }

        if (x_next_idx == -1) {
            num_open += 1;
        } else if (cell_indices[x_next_idx] == -1) {
            sum_open += input_concentration[x_next_idx];
            num_open += 1;
        }

        if (y_prev_idx == -1) {
            num_open += 1;
        } else if (cell_indices[y_prev_idx] == -1) {
            sum_open += input_concentration[y_prev_idx];
            num_open += 1;
        }

        if (y_next_idx == -1) {
            num_open += 1;
        } else if (cell_indices[y_next_idx] == -1) {
            sum_open += input_concentration[y_next_idx];
            num_open += 1;
        }

        if (z_prev_idx == -1) {
            num_open += 1;
        } else if (cell_indices[z_prev_idx] == -1) {
            sum_open += input_concentration[z_prev_idx];
            num_open += 1;
        }

        if (z_next_idx == -1) {
            num_open += 1;
        } else if (cell_indices[z_next_idx] == -1) {
            sum_open += input_concentration[z_next_idx];
            num_open += 1;
        }

        let diffusion_term = combined_constant * (sum_open - f32(num_open) * input_concentration[idx]);

        output_concentration[idx] += input_concentration[idx] + diffusion_term;
    }
}
`;