import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "prompt_relay.PowerLoraGateNative",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PromptRelayPowerLoraGate") {
            
            // Fungsi pembantu untuk menambahkan satu set widget LoRA
            const addLoraRow = function(node, index, loraName = "None") {
                const getLoraList = () => {
                    const loraLoader = LiteGraph.registered_node_types["LoraLoader"];
                    if (loraLoader && loraLoader.nodeData && loraLoader.nodeData.input && loraLoader.nodeData.input.required && loraLoader.nodeData.input.required.lora_name) {
                        return ["None"].concat(loraLoader.nodeData.input.required.lora_name[0] || []);
                    }
                    return ["None"];
                };

                node.addWidget("combo", `lora_${index}`, loraName, () => {}, { values: getLoraList() });
                node.addWidget("number", `segment_${index}`, 0, () => {}, { min: 0, max: 99, step: 10, precision: 0 });
                node.addWidget("number", `modelStr_${index}`, 1.0, () => {}, { min: -10.0, max: 10.0, step: 1, precision: 2 });
                node.addWidget("number", `clipStr_${index}`, 1.0, () => {}, { min: -10.0, max: 10.0, step: 1, precision: 2 });
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Pastikan widget dinamis mendapatkan izin untuk disimpan ke memori
                this.serialize_widgets = true;
                this.loraCount = 0;

                // Tombol: Tambah LoRA
                this.addWidget("button", "➕ Add LoRA", "add_lora", (value, widget, node, pos, event) => {
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
                
                // Tombol: Kurangi LoRA
                this.addWidget("button", "➖ Remove Last LoRA", "remove_lora", () => {
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

            // REVISI TOTAL: Pendekatan "Detektif Memori"
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                if (info && info.widgets_values) {
                    this.loraCount = 0;
                    
                    // Kita pindai metadata widget_values untuk mendeteksi blok LoRA secara akurat.
                    // Pola yang dicari: Teks (Nama), Angka (Segment), Angka (ModelStr), Angka (ClipStr)
                    let loraIndex = 1;
                    for (let i = 0; i < info.widgets_values.length; i++) {
                        const v1 = info.widgets_values[i];
                        const v2 = info.widgets_values[i+1];
                        const v3 = info.widgets_values[i+2];
                        const v4 = info.widgets_values[i+3];
                        
                        // Jika polanya cocok persis dengan widget kita
                        if (typeof v1 === "string" && typeof v2 === "number" && typeof v3 === "number" && typeof v4 === "number") {
                            this.loraCount = loraIndex;
                            addLoraRow(this, loraIndex, v1);
                            
                            // Suntikkan nilai ke widget yang baru saja dibuat
                            const len = this.widgets.length;
                            this.widgets[len - 3].value = v2; // Isi nilai segment
                            this.widgets[len - 2].value = v3; // Isi nilai modelStr
                            this.widgets[len - 1].value = v4; // Isi nilai clipStr
                            
                            loraIndex++;
                            i += 3; // Lompat ke kelompok data selanjutnya
                        }
                    }
                }

                // Lanjutkan fungsi bawaan ComfyUI
                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }
            };
        }
    }
});
