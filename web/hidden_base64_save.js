import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Comfy.RikanHiddenBase64ImageSaver",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RikanHiddenBase64ImageSaver") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                this.base64_result = "";

                // Tombol Copy Base64
                this.addWidget("button", "📋 Copy Base64", "copy", () => {
                    if (this.base64_result) {
                        navigator.clipboard.writeText(this.base64_result).then(() => {
                            alert("Base64 successfully copied to clipboard!");
                        });
                    } else {
                        alert("Base64 Data is not available. Run the workflow first.");
                    }
                });

                // Tombol Popup View
                this.addWidget("button", "🔍 View Decoded Image", "view", () => {
                    if (this.base64_result) {
                        let cleanB64 = this.base64_result;
                        if (cleanB64.includes(",")) cleanB64 = cleanB64.split(",")[1];
                        
                        const overlay = document.createElement("div");
                        overlay.style.cssText = "position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.85); z-index: 10000; display: flex; justify-content: center; align-items: center;";
                        
                        const modal = document.createElement("div");
                        modal.style.cssText = "background: #222; padding: 20px; border-radius: 10px; display: flex; flex-direction: column; align-items: center; max-width: 90vw; max-height: 90vh; box-shadow: 0 10px 25px rgba(0,0,0,0.8);";
                        modal.onclick = (e) => e.stopPropagation();

                        const img = document.createElement("img");
                        img.src = "data:image/png;base64," + cleanB64;
                        img.style.cssText = "max-width: 100%; max-height: 55vh; object-fit: contain; background: #000; border: 1px solid #444; border-radius: 6px;";
                        modal.appendChild(img);

                        // Kolom Textarea Read-Only untuk Saver
                        const textArea = document.createElement("textarea");
                        textArea.value = cleanB64;
                        textArea.readOnly = true;
                        textArea.style.cssText = "width: 100%; height: 100px; margin-top: 15px; background: #111; color: #4ade80; border: 1px solid #444; padding: 10px; font-family: monospace; font-size: 12px; resize: none; border-radius: 6px;";
                        textArea.onclick = () => textArea.select();
                        modal.appendChild(textArea);

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
                        alert("No data available yet. Please run the workflow first.");
                    }
                });

                return r;
            };

            // Menangkap output hasil generate dari Python
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (onExecuted) onExecuted.apply(this, arguments);

                if (message && message.base64_data && message.base64_data.length > 0) {
                    this.base64_result = message.base64_data[0];
                } else if (message && message.string && message.string.length > 0) {
                    this.base64_result = message.string[0];
                } else if (message && message.text && message.text.length > 0) {
                    this.base64_result = message.text[0];
                }
            };
        }
    }
});