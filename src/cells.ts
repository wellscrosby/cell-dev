export class Cell {
    constructor(public position: { x: number, y: number, z: number }, public radius: number) {
        this.position = position;
        this.radius = radius;
    }


}
