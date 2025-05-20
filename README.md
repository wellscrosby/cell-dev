# cell-dev

A simple webgpu/threejs cell development simulator with cells producing and responding to diffusion gradients.

## Install

`npm install`
`npx vite`

## Advice

-   Use Dedicated Laptop GPU (huge performance uplift):
    -   If using Chrome on windows, to use your laptop's dGPU you need to go to Settings>System>Display>Graphics, then click "Add desktop app" select chrome.exe (probably in C:\Program Files\Google\Chrome\Application) then scroll down to Google Chrome and change GPU preference to High Performance. This is a known issue that seems to be being worked on by the Chrome team (https://issues.chromium.org/issues/369219127).
