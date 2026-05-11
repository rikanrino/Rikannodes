import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "prompt_relay.MultiLoraGateNative",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RikanPromptRelayMultiLoraGate") {
            
            const syncData = (node) => {
                if (!node || !node.widgets) return;
                const dataWidget = node.widgets.find(w => w.name === "lora_data");
                if (!dataWidget) return;

                const loras = [];
                let maxIndex = 0;
                for (const w of node.widgets) {
                    if (w.name && w.name.startsWith("enable_")) {
                        const idx = parseInt(w.name.split("_")[1]);
                        if (idx > maxIndex) maxIndex = idx;
                    }
                }

                for (let i = 1; i <= maxIndex; i++) {
                    const wToggle = node.widgets.find(w => w.name === `enable_${i}`);
                    const wLora = node.widgets.find(w => w.name === `lora_${i}`);
                    const wSeg = node.widgets.find(w => w.name === `segment_${i}`);
                    const wMod = node.widgets.find(w => w.name === `modelStr_${i}`);

                    if (wToggle && wLora && wSeg && wMod) {
                        loras.push({
                            enable: wToggle.value,
                            name: wLora.value,
                            segment: wSeg.value,
                            modelStr: wMod.value
                        });
                    }
                }
                dataWidget.value = JSON.stringify(loras);
            };

            const addLoraRow = function(node, index, loraName = "None") {
                const getLoraList = () => {
                    const loraLoader = LiteGraph.registered_node_types["LoraLoader"];
                    if (loraLoader && loraLoader.nodeData && loraLoader.nodeData.input && loraLoader.nodeData.input.required && loraLoader.nodeData.input.required.lora_name) {
                        return ["None"].concat(loraLoader.nodeData.input.required.lora_name[0] || []);
                    }
                    return ["None"];
                };

                const wToggle = node.addWidget("toggle", `enable_${index}`, true, () => syncData(node));
                const wLora = node.addWidget("combo", `lora_${index}`, loraName, () => syncData(node), { values: getLoraList() });
                
                // LiteGraph membagi step dengan 10 untuk tombol klik.
                // step: 10 artinya tombol panah menambah 1. Default segment berurutan (0, 1, 2...)
                const defaultSeg = Math.max(0, index - 1);
                const wSeg = node.addWidget("number", `segment_${index}`, defaultSeg, () => syncData(node), { min: 0, max: 99, step: 10, precision: 0 });
                
                // step: 0.1 artinya tombol panah menambah 0.01. Default: 0.8
                const wMod = node.addWidget("number", `modelStr_${index}`, 0.80, () => syncData(node), { min: -10.0, max: 10.0, step: 0.1, precision: 2 });

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
                this.loraCount = 0;

                const wData = this.widgets.find(w => w.name === "lora_data");
                if (wData) {
                    wData.type = "hidden";
                    wData.hidden = true;
                    wData.computeSize = () => [0, -4];
                }

                this.btnToggleAll = this.addWidget("button", "🔄 Toggle All", "toggle_all", () => {
                    let allOn = true;
                    for (const w of this.widgets) {
                        if (w.name && w.name.startsWith("enable_") && w.value === false) {
                            allOn = false;
                            break;
                        }
                    }
                    const targetState = !allOn;
                    for (const w of this.widgets) {
                        if (w.name && w.name.startsWith("enable_")) {
                            w.value = targetState;
                        }
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
                            this.loraCount++;
                            addLoraRow(this, this.loraCount, loraName);
                            this.computeSize();
                            this.setDirtyCanvas(true, true);
                        }
                    }));

                    if (menuItems.length === 0) {
                        menuItems.push({ content: "No LoRAs found!", callback: () => {} });
                    }

                    const e = event || app.canvas.last_mouse_event;
                    new LiteGraph.ContextMenu(menuItems, { event: e, title: "Choose a LoRA" });
                });
                
                this.btnRemoveLora = this.addWidget("button", "➖ Remove Last LoRA", "remove_lora", () => {
                    if (this.loraCount > 0) {
                        const suffix = `_${this.loraCount}`;
                        for (let i = this.widgets.length - 1; i >= 0; i--) {
                            if (this.widgets[i].name && this.widgets[i].name.endsWith(suffix)) {
                                this.widgets.splice(i, 1);
                            }
                        }
                        this.loraCount--;
                        syncData(this);
                        this.computeSize();
                        this.setDirtyCanvas(true, true);
                    }
                });

                return r;
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                if (info && info.widgets_values) {
                    this.loraCount = 0;
                    
                    let jsonData = "[]";
                    for(let v of info.widgets_values) {
                        if (typeof v === "string" && v.startsWith("[") && v.endsWith("]")) {
                            jsonData = v;
                            break;
                        }
                    }

                    try {
                        const savedLoras = JSON.parse(jsonData);
                        for (let i = 0; i < savedLoras.length; i++) {
                            const lora = savedLoras[i];
                            const idx = i + 1;
                            this.loraCount = idx;
                            addLoraRow(this, idx, lora.name);
                            
                            const wToggle = this.widgets.find(w => w.name === `enable_${idx}`);
                            const wSegment = this.widgets.find(w => w.name === `segment_${idx}`);
                            const wModel = this.widgets.find(w => w.name === `modelStr_${idx}`);

                            if (wToggle) wToggle.value = lora.enable;
                            if (wSegment) wSegment.value = lora.segment;
                            if (wModel) wModel.value = lora.modelStr;
                        }
                        
                        const wData = this.widgets.find(w => w.name === "lora_data");
                        if (wData) {
                            wData.type = "hidden";
                            wData.hidden = true;
                            wData.computeSize = () => [0, -4];
                            wData.value = jsonData;
                        }

                    } catch (e) {
                        console.error("[Rikan MultiLoraGate] Failed to parse saved JSON data:", e);
                    }
                }

                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }
            };
        }
    }
});