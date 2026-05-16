import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "rikan.MultiLoraLoaderNative", // Ubah nama internal ekstensi
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // PERBAIKAN: Menargetkan nama node yang baru
        if (nodeData.name === "RikanMultiLoraLoader") { 
            
            const syncData = (node) => {
                const dataWidget = node.widgets.find(w => w.name === "lora_data");
                if (!dataWidget) return;

                const loras = [];
                const loraWidgets = node.widgets.filter(w => w.type === "RIKAN_LORA_LOADER_ROW");
                
                loraWidgets.forEach(w => {
                    loras.push({
                        enable: w.value.enable,
                        name: w.value.name,
                        modelStr: w.value.modelStr
                    });
                });
                dataWidget.value = JSON.stringify(loras);
            };

            const showInfoPopup = (data) => {
                const overlay = document.createElement("div");
                overlay.style.cssText = "position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.85); z-index: 10000; display: flex; justify-content: center; align-items: center; font-family: sans-serif;";
                
                const modal = document.createElement("div");
                modal.style.cssText = "background: #2b2b2b; width: 900px; max-height: 90vh; border-radius: 8px; box-shadow: 0 10px 30px rgba(0,0,0,0.8); display: flex; flex-direction: column; overflow: hidden; border: 1px solid #444; color: #ddd;";
                modal.onclick = (e) => e.stopPropagation();

                let wordsHtml = "<em>None</em>";
                if (data.trained_words && data.trained_words.length > 0) {
                    wordsHtml = data.trained_words.map(w => 
                        `<span style="background: #1e3a8a; color: #93c5fd; padding: 2px 8px; border-radius: 4px; margin-right: 6px; font-size: 12px; font-weight: bold; border: 1px solid #2563eb; cursor: pointer;" onclick="navigator.clipboard.writeText('${w}'); alert('Copied: ${w}')" title="Click to copy">${w}</span>`
                    ).join('');
                }

                const infoSection = document.createElement("div");
                infoSection.style.cssText = "padding: 20px; border-bottom: 1px solid #444;";
                infoSection.innerHTML = `
                    <h2 style="margin: 0 0 15px 0; color: #fff; font-size: 20px;">${data.model_name || "Unknown Model"}</h2>
                    <div style="display: flex; gap: 10px; margin-bottom: 20px;">
                        <span style="background: #6b21a8; color: #d8b4fe; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold;">LORA</span>
                        <span style="background: #3f3f46; color: #a1a1aa; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold;">${data.base_model || "Unknown Base"}</span>
                    </div>
                    
                    <table style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: left;">
                        <tr style="border-top: 1px solid #444;"><td style="padding: 8px; width: 120px; color: #aaa;">File</td><td style="padding: 8px;">${data.filename || "-"}</td></tr>
                        <tr style="border-top: 1px solid #444;"><td style="padding: 8px; color: #aaa;">Hash (SHA256)</td><td style="padding: 8px; font-family: monospace;">${data.hash || "-"}</td></tr>
                        <tr style="border-top: 1px solid #444;"><td style="padding: 8px; color: #aaa;">Civitai</td><td style="padding: 8px;"><a href="${data.url || '#'}" target="_blank" style="color: #60a5fa; text-decoration: none;">🌐 View on Civitai</a></td></tr>
                        <tr style="border-top: 1px solid #444; border-bottom: 1px solid #444;"><td style="padding: 8px; color: #aaa;">Trained Words</td><td style="padding: 8px;">${wordsHtml}</td></tr>
                    </table>
                `;
                modal.appendChild(infoSection);

                const mediaSection = document.createElement("div");
                mediaSection.style.cssText = "padding: 20px; overflow-y: auto; display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; background: #1f1f1f;";
                
                if (data.media && data.media.length > 0) {
                    data.media.forEach(m => {
                        let mediaEl = m.type === "video" ? 
                            `<video src="${m.url}" autoplay loop muted controls style="width: 100%; height: 350px; object-fit: cover; background: #000; display: block;"></video>` : 
                            `<img src="${m.url}" style="width: 100%; height: 350px; object-fit: cover; background: #000; display: block;">`;

                        let promptHtml = m.prompt ? `
                            <div style="margin-top: 8px; border: 1px solid #555; border-radius: 4px; padding: 6px; background: #111; max-height: 80px; overflow-y: auto; font-size: 11px; line-height: 1.4; color: #e5e7eb; cursor: text;">
                                <span style="color: #9ca3af; margin-right: 5px; user-select: none;">positive</span>${m.prompt}
                            </div>
                        ` : "";

                        const overlayHtml = `
                            <div class="media-overlay" style="position: absolute; bottom: 0; left: 0; width: 100%; background: rgba(35,35,35,0.95); padding: 12px; box-sizing: border-box; opacity: 0; transition: opacity 0.2s ease-in-out; border-top: 1px solid #555; pointer-events: auto;">
                                <a href="${m.page_url || m.url}" target="_blank" style="display: inline-block; color: #fff; text-decoration: none; border: 1px solid #777; background: #000; padding: 3px 8px; border-radius: 4px; font-size: 11px;">civitai 🔗</a>
                                ${promptHtml}
                            </div>
                        `;

                        const containerHtml = `
                            <div style="position: relative; border-radius: 6px; overflow: hidden; border: 1px solid #333;" 
                                 onmouseenter="this.querySelector('.media-overlay').style.opacity=1" 
                                 onmouseleave="this.querySelector('.media-overlay').style.opacity=0">
                                ${mediaEl}
                                ${overlayHtml}
                            </div>
                        `;
                        mediaSection.innerHTML += containerHtml;
                    });
                } else {
                    mediaSection.innerHTML = `<p style="color: #777; width: 100%; text-align: center;">No preview media available.</p>`;
                }
                modal.appendChild(mediaSection);

                const footer = document.createElement("div");
                footer.style.cssText = "padding: 15px; text-align: center; border-top: 1px solid #444; background: #2b2b2b;";
                const closeBtn = document.createElement("button");
                closeBtn.innerText = "Close";
                closeBtn.style.cssText = "padding: 8px 30px; background: #444; color: white; border: 1px solid #555; border-radius: 4px; cursor: pointer; font-size: 14px;";
                closeBtn.onmouseover = () => closeBtn.style.background = "#555";
                closeBtn.onmouseout = () => closeBtn.style.background = "#444";
                
                const destroyPopup = () => overlay.remove();
                closeBtn.onclick = destroyPopup;
                overlay.onclick = destroyPopup;
                
                footer.appendChild(closeBtn);
                modal.appendChild(footer);
                overlay.appendChild(modal);
                document.body.appendChild(overlay);
            };

            const addLoraRow = function(node, loraData) {
                const w = node.addWidget("custom", "lora_row_" + Math.random().toString(36).substr(2, 5), loraData, () => syncData(node));
                w.type = "RIKAN_LORA_LOADER_ROW"; 
                w.civitaiData = null;

                w.fetchCivitaiData = function(name) {
                    if (name === "None" || !name) {
                        this.civitaiData = null;
                        return;
                    }
                    this.civitaiData = { loading: true };
                    node.setDirtyCanvas(true, true);

                    fetch("/rikan/get_lora_info", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({lora_name: name})
                    })
                    .then(r => r.json())
                    .then(data => {
                        this.civitaiData = data;
                        node.setDirtyCanvas(true, true);
                    }).catch(err => {
                        this.civitaiData = null;
                        node.setDirtyCanvas(true, true);
                    });
                };
                
                w.fetchCivitaiData(loraData.name);

                w.computeSize = function(width) {
                    return [width, 28];
                };

                w.draw = function(ctx, node, width, y, margin) {
                    this.last_y = y; 
                    
                    ctx.save();
                    const v = this.value;
                    const isEnabled = v.enable;
                    const fgColor = isEnabled ? "#ddd" : "#666";
                    const bgColor = isEnabled ? "#262626" : "#1e1e1e";

                    // Background
                    ctx.fillStyle = bgColor;
                    ctx.beginPath();
                    ctx.roundRect(margin, y + 2, width - margin * 2, 24, 6);
                    ctx.fill();

                    // Toggle Switch
                    const togX = margin + 10;
                    ctx.fillStyle = isEnabled ? "#4285f4" : "#444";
                    ctx.beginPath();
                    ctx.roundRect(togX, y + 7, 24, 14, 7);
                    ctx.fill();
                    ctx.fillStyle = "#fff";
                    ctx.beginPath();
                    ctx.arc(isEnabled ? togX + 17 : togX + 7, y + 14, 5, 0, Math.PI * 2);
                    ctx.fill();

                    // Info Button
                    const infoX = width - margin - 75;
                    if (this.civitaiData && this.civitaiData.loading) {
                        ctx.fillStyle = "#b45309";
                        ctx.beginPath(); ctx.roundRect(infoX, y + 6, 16, 16, 3); ctx.fill();
                        ctx.fillStyle = "#fff"; ctx.font = "bold 11px Arial"; ctx.fillText("⏳", infoX + 2, y + 18);
                    } else if (this.civitaiData && this.civitaiData.found) {
                        ctx.fillStyle = "#1d4ed8";
                        ctx.beginPath(); ctx.roundRect(infoX, y + 6, 16, 16, 3); ctx.fill();
                        ctx.fillStyle = "#fff"; ctx.font = "bold 12px Arial"; ctx.fillText("i", infoX + 6, y + 18);
                    } else {
                        ctx.fillStyle = "#444";
                        ctx.beginPath(); ctx.roundRect(infoX, y + 6, 16, 16, 3); ctx.fill();
                        ctx.fillStyle = "#888"; ctx.font = "bold 12px Arial"; ctx.fillText("x", infoX + 4, y + 18);
                    }

                    // Strength
                    const strX = width - margin - 45;
                    ctx.fillStyle = isEnabled ? "#111" : "#1a1a1a";
                    ctx.beginPath(); ctx.roundRect(strX, y + 5, 40, 18, 3); ctx.fill();
                    ctx.fillStyle = fgColor; ctx.textAlign = "center"; ctx.font = "11px Arial";
                    ctx.fillText(v.modelStr.toFixed(2), strX + 20, y + 18);

                    // LoRA Name
                    ctx.textAlign = "left";
                    ctx.fillStyle = fgColor;
                    ctx.font = "13px Arial";
                    let displayName = v.name;
                    
                    const maxTextWidth = (infoX - 10) - (togX + 35);
                    if (ctx.measureText(displayName).width > maxTextWidth) {
                        while (displayName.length > 0 && ctx.measureText(displayName + "...").width > maxTextWidth) {
                            displayName = displayName.slice(0, -1);
                        }
                        displayName += "...";
                    }
                    ctx.fillText(displayName, togX + 35, y + 18);

                    ctx.restore();
                };

                w.mouse = function(event, pos, node) {
                    if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
                    
                    const [x, y] = pos;
                    const wY = this.last_y || 0; 
                    
                    if (y < wY || y > wY + 28) return false;

                    const width = node.size[0];
                    const margin = LiteGraph.NODE_WIDGET_MARGIN || 15;
                    const v = this.value;

                    // Toggle
                    if (x >= margin + 10 && x <= margin + 35) {
                        v.enable = !v.enable;
                        syncData(node);
                        node.setDirtyCanvas(true, true);
                        return true;
                    }

                    // Info Button
                    const infoX = width - margin - 75;
                    if (x >= infoX && x <= infoX + 16) {
                        if (this.civitaiData && this.civitaiData.found) {
                            showInfoPopup(this.civitaiData);
                        } else if (this.civitaiData && !this.civitaiData.found) {
                            alert("This LoRA is not found on Civitai's Database.");
                        }
                        return true;
                    }

                    // Strength
                    const strX = width - margin - 45;
                    if (x >= strX && x <= strX + 40) {
                        app.canvas.prompt("Set Strength (-10.0 to 10.0):", v.modelStr, (val) => {
                            const parsed = parseFloat(val);
                            if (!isNaN(parsed)) {
                                v.modelStr = parsed;
                                syncData(node);
                                node.setDirtyCanvas(true, true);
                            }
                        }, event);
                        return true;
                    }

                    // Name Dropdown
                    if (x > margin + 35 && x < infoX - 5) {
                        const loraLoader = LiteGraph.registered_node_types["LoraLoader"];
                        let loras = ["None"];
                        if (loraLoader && loraLoader.nodeData && loraLoader.nodeData.input && loraLoader.nodeData.input.required && loraLoader.nodeData.input.required.lora_name) {
                            loras = ["None"].concat(loraLoader.nodeData.input.required.lora_name[0] || []);
                        }
                        const menuItems = loras.map(l => ({
                            content: l,
                            callback: () => {
                                v.name = l;
                                this.fetchCivitaiData(l);
                                syncData(node);
                                node.setDirtyCanvas(true, true);
                            }
                        }));
                        new LiteGraph.ContextMenu(menuItems, { event: event, title: "Select LoRA" });
                        return true;
                    }

                    return false;
                };

                if (node.btnAddLora && node.btnRemoveLora) {
                    const idx2 = node.widgets.indexOf(node.btnAddLora);
                    if (idx2 !== -1) node.widgets.splice(idx2, 1);
                    const idx3 = node.widgets.indexOf(node.btnRemoveLora);
                    if (idx3 !== -1) node.widgets.splice(idx3, 1);
                    node.widgets.push(node.btnAddLora);
                    node.widgets.push(node.btnRemoveLora);
                }
                syncData(node);
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                this.serialize_widgets = true;

                const wData = this.widgets.find(w => w.name === "lora_data");
                if (wData) {
                    wData.type = "hidden";
                    wData.hidden = true;
                    wData.computeSize = () => [0, -4];
                }

                this.btnToggleAll = this.addWidget("button", "🔄 Toggle All Enable/Disable", "toggle_all", () => {
                    let allOn = true;
                    const loraWidgets = this.widgets.filter(w => w.type === "RIKAN_LORA_LOADER_ROW");
                    for (const w of loraWidgets) {
                        if (w.value.enable === false) { allOn = false; break; }
                    }
                    const targetState = !allOn;
                    for (const w of loraWidgets) {
                        w.value.enable = targetState;
                    }
                    syncData(this);
                    this.setDirtyCanvas(true, true);
                });

                this.btnAddLora = this.addWidget("button", "➕ Add LoRA", "add_lora", (value, widget, node, pos, event) => {
                    const loraLoader = LiteGraph.registered_node_types["LoraLoader"];
                    let loras = ["None"];
                    if (loraLoader && loraLoader.nodeData && loraLoader.nodeData.input && loraLoader.nodeData.input.required && loraLoader.nodeData.input.required.lora_name) {
                        loras = loraLoader.nodeData.input.required.lora_name[0].filter(l => l !== "None");
                    }
                    
                    const menuItems = loras.map(loraName => ({
                        content: loraName,
                        callback: () => {
                            addLoraRow(this, { enable: true, name: loraName, modelStr: 1.00 });
                            this.computeSize();
                            this.setDirtyCanvas(true, true);
                        }
                    }));

                    if (menuItems.length === 0) menuItems.push({ content: "No LoRAs found!", callback: () => {} });
                    
                    const e = event || app.canvas.last_mouse_event;
                    new LiteGraph.ContextMenu(menuItems, { event: e, title: "Choose a LoRA" });
                });
                
                this.btnRemoveLora = this.addWidget("button", "➖ Remove Last LoRA", "remove_lora", () => {
                    for (let i = this.widgets.length - 1; i >= 0; i--) {
                        if (this.widgets[i].type === "RIKAN_LORA_LOADER_ROW") {
                            this.widgets.splice(i, 1);
                            syncData(this);
                            this.computeSize();
                            this.setDirtyCanvas(true, true);
                            break;
                        }
                    }
                });

                return r;
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                if (info && info.widgets_values) {
                    let jsonData = "[]";
                    for(let v of info.widgets_values) {
                        if (typeof v === "string" && v.startsWith("[") && v.endsWith("]")) {
                            jsonData = v;
                            break;
                        }
                    }

                    try {
                        for (let i = this.widgets.length - 1; i >= 0; i--) {
                            if (this.widgets[i].type === "RIKAN_LORA_LOADER_ROW" || this.widgets[i].name.startsWith("enable_")) {
                                this.widgets.splice(i, 1);
                            }
                        }

                        const savedLoras = JSON.parse(jsonData);
                        for (let i = 0; i < savedLoras.length; i++) {
                            const lora = savedLoras[i];
                            addLoraRow(this, { enable: lora.enable, name: lora.name, modelStr: lora.modelStr });
                        }
                        
                        const wData = this.widgets.find(w => w.name === "lora_data");
                        if (wData) {
                            wData.type = "hidden";
                            wData.hidden = true;
                            wData.computeSize = () => [0, -4];
                            wData.value = jsonData;
                        }
                    } catch (e) {
                        console.error("[Rikan MultiLoraLoader] Failed to parse saved JSON data:", e);
                    }
                }

                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }
            };
            
            const onSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function (o) {
                if (onSerialize) onSerialize.apply(this, arguments);
                syncData(this);
            };
        }
    }
});