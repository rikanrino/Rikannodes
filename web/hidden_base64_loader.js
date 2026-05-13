import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Comfy.RikanHiddenBase64ImageLoader",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RikanHiddenBase64ImageLoader") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // 1. Setup UI Textbox Base64
                const base64Widget = this.widgets.find((w) => w.name === "base64_data");
                if (base64Widget) {
                    base64Widget.computeSize = function() { return [this.width || 200, 85]; };
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

                // 2. Setup Upload File
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
                            const rawBase64 = event.target.result.split(',')[1];
                            if (base64Widget) base64Widget.value = rawBase64;
                        };
                        reader.readAsDataURL(file);
                    }
                });

                this.addWidget("button", "Upload Image (Local Blob)", "upload", () => {
                    fileInput.click();
                });

                // 3. Tombol Popup View
                this.addWidget("button", "🔍 View Decoded Image", "view", () => {
                    if (base64Widget && base64Widget.value) {
                        // Jalankan fungsi popup dari dalam
                        let cleanB64 = base64Widget.value;
                        if (cleanB64.includes(",")) cleanB64 = cleanB64.split(",")[1];
                        
                        const overlay = document.createElement("div");
                        overlay.style.cssText = "position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.85); z-index: 10000; display: flex; justify-content: center; align-items: center;";
                        
                        const modal = document.createElement("div");
                        modal.style.cssText = "background: #222; padding: 20px; border-radius: 10px; display: flex; flex-direction: column; align-items: center; max-width: 90vw; max-height: 90vh; box-shadow: 0 10px 25px rgba(0,0,0,0.8);";
                        modal.onclick = (e) => e.stopPropagation();

                        const img = document.createElement("img");
                        img.src = "data:image/png;base64," + cleanB64;
                        img.style.cssText = "max-width: 100%; max-height: 70vh; object-fit: contain; background: #000; border: 1px solid #444; border-radius: 6px;";
                        modal.appendChild(img);

                        const closeBtn = document.createElement("button");
                        closeBtn.innerText = "Close & Clear Memory";
                        closeBtn.style.cssText = "margin-top: 15px; padding: 10px; background: #ef4444; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; width: 100%; font-size: 14px;";
                        
                        const destroyPopup = () => { img.src = ""; overlay.remove(); };
                        closeBtn.onclick = destroyPopup;
                        overlay.onclick = destroyPopup;

                        modal.appendChild(closeBtn);
                        overlay.appendChild(modal);
                        document.body.appendChild(overlay);
                    } else {
                        alert("The form is empty. Please paste a Base64 string or upload an image first.");
                    }
                });

                return r;
            };
        }
    }
});