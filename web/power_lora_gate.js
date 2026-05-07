import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "prompt_relay.PowerLoraGateNative",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // PERHATIKAN PERUBAHAN NAMA DI BAWAH INI
        if (nodeData.name === "RikanPromptRelayPowerLoraGate") {
            
            const addLoraRow = function(node, index, loraName = "None") {
                const getLoraList = () => {
                    const loraLoader = LiteGraph.registered_node_types["LoraLoader"];
                    if (loraLoader && loraLoader.nodeData && loraLoader.nodeData.input && loraLoader.nodeData.input.required && loraLoader.nodeData.input.required.lora_name) {
                        return ["None"].concat(loraLoader.nodeData.input.required.lora_name[0] || []);
                    }
                    return ["None"];
                };

                node.addWidget("toggle", `enable_${index}`, true, () => {});
                node.addWidget("combo", `lora_${index}`, loraName, () => {}, { values: getLoraList() });
                node.addWidget("number", `segment_${index}`, 1, () => {}, { min: 1, max: 100, step: 10, precision: 0 });
                node.addWidget("number", `modelStr_${index}`, 1.0, () => {}, { min: -10.0, max: 10.0, step: 1, precision: 2 });
                node.addWidget("number", `clipStr_${index}`, 1.0, () => {}, { min: -10.0, max: 10.0, step: 1, precision: 2 });

                if (node.btnAddLora && node.btnRemoveLora) {
                    const idx2 = node.widgets.indexOf(node.btnAddLora);
                    if (idx2 !== -1) node.widgets.splice(idx2, 1);
                    
                    const idx3 = node.widgets.indexOf(node.btnRemoveLora);
                    if (idx3 !== -1) node.widgets.splice(idx3, 1);

                    node.widgets.push(node.btnAddLora);
                    node.widgets.push(node.btnRemoveLora);
                }
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                this.serialize_widgets = true;
                this.loraCount = 0;

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
                    new LiteGraph.ContextMenu(menuItems, { event: e, title: "Choose a lora" });
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
                    let loraIndex = 1;
                    
                    for (let i = 0; i < info.widgets_values.length; i++) {
                        const v1 = info.widgets_values[i];
                        
                        // Deteksi pola BARU
                        if (typeof v1 === "boolean" && 
                            typeof info.widgets_values[i+1] === "string" && 
                            typeof info.widgets_values[i+2] === "number" && 
                            typeof info.widgets_values[i+3] === "number" && 
                            typeof info.widgets_values[i+4] === "number") {
                            
                            this.loraCount = loraIndex;
                            addLoraRow(this, loraIndex, info.widgets_values[i+1]);
                            
                            const wToggle = this.widgets.find(w => w.name === `enable_${loraIndex}`);
                            const wSegment = this.widgets.find(w => w.name === `segment_${loraIndex}`);
                            const wModel = this.widgets.find(w => w.name === `modelStr_${loraIndex}`);
                            const wClip = this.widgets.find(w => w.name === `clipStr_${loraIndex}`);

                            if (wToggle) wToggle.value = v1; 
                            if (wSegment) wSegment.value = Math.max(1, info.widgets_values[i+2]); 
                            if (wModel) wModel.value = info.widgets_values[i+3]; 
                            if (wClip) wClip.value = info.widgets_values[i+4]; 
                            
                            loraIndex++;
                            i += 4; 
                        }
                        // Deteksi pola LAMA
                        else if (typeof v1 === "string" && 
                                 typeof info.widgets_values[i+1] === "number" && 
                                 typeof info.widgets_values[i+2] === "number" && 
                                 typeof info.widgets_values[i+3] === "number") {
                            
                            this.loraCount = loraIndex;
                            addLoraRow(this, loraIndex, v1);
                            
                            const wToggle = this.widgets.find(w => w.name === `enable_${loraIndex}`);
                            const wSegment = this.widgets.find(w => w.name === `segment_${loraIndex}`);
                            const wModel = this.widgets.find(w => w.name === `modelStr_${loraIndex}`);
                            const wClip = this.widgets.find(w => w.name === `clipStr_${loraIndex}`);

                            if (wToggle) wToggle.value = true;
                            if (wSegment) wSegment.value = Math.max(1, info.widgets_values[i+1]); 
                            if (wModel) wModel.value = info.widgets_values[i+2]; 
                            if (wClip) wClip.value = info.widgets_values[i+3]; 
                            
                            loraIndex++;
                            i += 3;
                        }
                    }
                }

                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }
            };
        }
    }
});