import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Comfy.RikanHiddenBase64ImageLoader",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RikanHiddenBase64ImageLoader") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                this.show_custom_image = false;
                this.custom_image_element = null;

                const base64Widget = this.widgets.find((w) => w.name === "base64_data");
                if (base64Widget) {
                    // Memberikan ruang 85px di LiteGraph (60px untuk text area + 25px untuk margin bawah)
                    base64Widget.computeSize = function() {
                        return [this.width || 200, 85];
                    };
                    
                    // Tinggi elemen fisik tetap 60px agar rapi
                    if (base64Widget.inputEl) {
                        base64Widget.inputEl.style.setProperty("height", "60px", "important");
                        base64Widget.inputEl.style.setProperty("max-height", "60px", "important");
                        base64Widget.inputEl.style.setProperty("min-height", "60px", "important");
                        base64Widget.inputEl.style.setProperty("resize", "none", "important");
                        base64Widget.inputEl.style.overflowY = "scroll";
                        base64Widget.inputEl.style.overflowX = "hidden";
                        base64Widget.inputEl.style.wordBreak = "break-all";
                    }
                }

                const fileInput = document.createElement("input");
                fileInput.type = "file";
                fileInput.accept = "image/*";
                fileInput.style.display = "none";
                document.body.appendChild(fileInput);

                fileInput.addEventListener("change", (e) => {
                    const file = e.target.files[0];
                    if (file) {
                        const reader = new FileReader();
                        reader.onload = (event) => {
                            const dataUri = event.target.result;
                            const rawBase64 = dataUri.split(',')[1];
                            
                            if (base64Widget) {
                                base64Widget.value = rawBase64;
                            }
                            
                            this.custom_image_element = new Image();
                            this.custom_image_element.src = dataUri; 
                            this.custom_image_element.onload = () => {
                                this.size[1] = this.computeSize()[1];
                                this.setDirtyCanvas(true, true);
                            };
                        };
                        reader.readAsDataURL(file);
                    }
                });

                this.addWidget("button", "Upload Image", "upload", () => {
                    fileInput.click();
                });

                requestAnimationFrame(() => {
                    this.size[1] = this.computeSize()[1];
                    this.setDirtyCanvas(true, true);
                });

                setTimeout(() => {
                    if (base64Widget && base64Widget.value) {
                        this.custom_image_element = new Image();
                        this.custom_image_element.src = "data:image/png;base64," + base64Widget.value;
                        this.custom_image_element.onload = () => {
                            this.setDirtyCanvas(true, true);
                        };
                    }
                }, 100);

                return r;
            };

            const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function (_, options) {
                if (getExtraMenuOptions) {
                    getExtraMenuOptions.apply(this, arguments);
                }

                options.unshift({
                    content: this.show_custom_image ? "Hide Image" : "Show Image",
                    callback: () => {
                        this.show_custom_image = !this.show_custom_image;
                        
                        if (!this.show_custom_image) {
                            this.size[1] = this.computeSize()[1];
                        }
                        
                        this.setDirtyCanvas(true, true);
                    }
                });
            };

            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                if (onDrawForeground) {
                    onDrawForeground.apply(this, arguments);
                }

                if (this.show_custom_image && this.custom_image_element) {
                    const padding = 10;
                    const w = this.size[0] - padding * 2;
                    
                    let y = 30; 
                    
                    const btnWidget = this.widgets ? this.widgets.find(w => w.name === "upload" || w.type === "button") : null;
                    if (btnWidget && btnWidget.last_y) {
                        y = btnWidget.last_y + 30; 
                    } else if (this.widgets && this.widgets.length > 0) {
                        y = this.widgets[this.widgets.length - 1].last_y + 30;
                    }

                    const imgW = this.custom_image_element.width;
                    const imgH = this.custom_image_element.height;
                    
                    const scale = w / imgW; 
                    const drawW = w;
                    const drawH = imgH * scale;

                    ctx.drawImage(this.custom_image_element, padding, y, drawW, drawH);
                    
                    const requiredHeight = y + drawH + padding;
                    if (this.size[1] < requiredHeight) {
                        this.size[1] = requiredHeight;
                    }
                }
            };
        }
    }
});