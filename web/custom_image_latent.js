import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Comfy.RikanQwenCustomImageSize.DynamicUI",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RikanQwenCustomImageSize") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                const resizeWidget = this.widgets?.find(w => w.name === "resize_mode");
                const textWidget = this.widgets?.find(w => w.name === "text");
                const presetWidget = this.widgets?.find(w => w.name === "resolution_preset");
                const orientWidget = this.widgets?.find(w => w.name === "orientation");
                const widthWidget = this.widgets?.find(w => w.name === "custom_width");
                const heightWidget = this.widgets?.find(w => w.name === "custom_height");

                if (textWidget && textWidget.inputEl) {
                    textWidget.inputEl.placeholder = "[WRITE YOUR PROMPT HERE]\n\nThe selected theme's addon will be appended automatically.";
                }

                const allWidgets = [...this.widgets];

                // Menampilkan UI jika mode membutuhkan prompt tambahan
                const toggleVisibility = () => {
                    const isAdvancedMode = resizeWidget.value === "Generative Fill" || resizeWidget.value === "Zoom Out";
                    
                    this.widgets = allWidgets.filter(w => {
                        if (w.name === "text" || w.name === "prompt_theme") {
                            return isAdvancedMode;
                        }
                        return true;
                    });

                    if (textWidget.inputEl) {
                        textWidget.inputEl.style.display = isAdvancedMode ? "block" : "none";
                    }

                    this.onResize?.(this.size);
                    this.setDirtyCanvas(true, true);
                };

                // Memperbarui angka resolusi otomatis
                const updateDimensions = () => {
                    if (presetWidget.value === "Custom") return;
                    let bL = 1280, bS = 720;
                    if (presetWidget.value === "512p (SD 1.5)") { bL = 768; bS = 512; }
                    else if (presetWidget.value === "720p (HD)") { bL = 1280; bS = 720; }
                    else if (presetWidget.value === "1080p (FHD)") { bL = 1920; bS = 1080; }
                    else if (presetWidget.value === "1024p (SDXL)") { bL = 1344; bS = 768; }

                    if (orientWidget.value === "Horizontal (Landscape)") {
                        widthWidget.value = bL; heightWidget.value = bS;
                    } else if (orientWidget.value === "Vertical (Portrait)") {
                        widthWidget.value = bS; heightWidget.value = bL;
                    } else {
                        widthWidget.value = bS; heightWidget.value = bS;
                    }
                };

                if (resizeWidget) resizeWidget.callback = () => toggleVisibility();
                if (presetWidget) presetWidget.callback = () => updateDimensions();
                if (orientWidget) orientWidget.callback = () => updateDimensions();

                setTimeout(() => {
                    toggleVisibility();
                    updateDimensions();
                }, 100);

                return r;
            };
        }
    }
});