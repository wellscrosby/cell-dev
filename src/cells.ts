import * as THREE from 'three';

export class Cell {
    public mesh: THREE.Mesh;

    constructor(public position: { x: number, y: number, z: number }, public productionRate: number) {
        this.position = position;
        const radius = 1; // Fixed radius instead of parameter

        const geometry = new THREE.SphereGeometry(radius, 32, 32);
        const material = new THREE.MeshBasicMaterial({
            color: 0xf00000,
            transparent: true,
            opacity: 0.2,
            depthTest: true,
            depthWrite: false
        });

        this.mesh = new THREE.Mesh(geometry, material);
        this.mesh.position.set(position.x + 0.5, position.y + 0.5, position.z + 0.5);
        this.mesh.renderOrder = 1;

        this.productionRate = productionRate;
    }
}

export function checkIntersection(cells: Cell[], position: { x: number, y: number, z: number }): boolean {
    for (const cell of cells) {
        if (cell.position.x == position.x && cell.position.y == position.y && cell.position.z == position.z) {
            return true;
        }
    }
    return false;
}
