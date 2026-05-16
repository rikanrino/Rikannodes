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
                const themeWidget = this.widgets?.find(w => w.name === "prompt_theme");
                const presetWidget = this.widgets?.find(w => w.name === "resolution_preset");
                const orientWidget = this.widgets?.find(w => w.name === "orientation");
                const widthWidget = this.widgets?.find(w => w.name === "custom_width");
                const heightWidget = this.widgets?.find(w => w.name === "custom_height");

                if (textWidget && textWidget.inputEl) {
                    textWidget.inputEl.placeholder = "[WRITE YOUR PROMPT HERE]\n\nThe selected theme's addon will be appended automatically.";
                }

                // Simpan opsi dan tipe asli ke dalam memori untuk direstorasi nanti
                if (themeWidget) {
                    themeWidget.origType = themeWidget.type;
                    themeWidget.origComputeSize = themeWidget.computeSize;
                }
                const origOrientOptions = orientWidget ? [...orientWidget.options.values] : [];

                // 1. Fungsi Show/Hide Widget (DIPERBAIKI)
                const toggleVisibility = () => {
                    const isAdvancedMode = resizeWidget.value === "Generative Fill" || resizeWidget.value === "Zoom Out";
                    
                    if (themeWidget) {
                        if (isAdvancedMode) {
                            // Munculkan kembali widget secara normal
                            themeWidget.type = themeWidget.origType;
                            themeWidget.computeSize = themeWidget.origComputeSize;
                        } else {
                            // Sembunyikan widget secara aman tanpa menghapusnya dari data node
                            themeWidget.type = "hidden";
                            themeWidget.computeSize = () => [0, -4]; // Ukuran negatif agar tidak menyisakan ruang kosong/gap di node
                        }
                    }

                    // Teks area selalu dipaksa muncul
                    if (textWidget && textWidget.inputEl) {
                        textWidget.inputEl.style.display = "block";
                    }

                    // Paksa node menghitung ulang tingginya agar rapi
                    if (this.computeSize) {
                        this.size[1] = this.computeSize()[1];
                    }
                    this.setDirtyCanvas(true, true);
                };

                // 2. Fungsi: Kunci Orientasi jika Full Body
                const checkFullBodyLock = () => {
                    if (!orientWidget || !themeWidget || !resizeWidget) return;
                    
                    const isAdvancedMode = resizeWidget.value === "Generative Fill" || resizeWidget.value === "Zoom Out";

                    if (themeWidget.value === "Full Body" && isAdvancedMode) {
                        // Paksa nilai menjadi Vertical
                        orientWidget.value = "Vertical (Portrait)";
                        // Kunci dropdown dengan hanya menyisakan 1 opsi
                        orientWidget.options.values = ["Vertical (Portrait)"];
                    } else {
                        // Kembalikan semua opsi jika mode lain dipilih
                        orientWidget.options.values = origOrientOptions;
                    }
                };

                // 3. Fungsi Kontrol Angka Resolusi
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

                // Memasang Listener ke setiap menu dropdown
                if (resizeWidget) {
                    resizeWidget.callback = () => {
                        toggleVisibility();
                        checkFullBodyLock();
                        updateDimensions();
                    };
                }
                if (themeWidget) {
                    themeWidget.callback = () => {
                        checkFullBodyLock();
                        updateDimensions();
                    };
                }
                if (presetWidget) {
                    presetWidget.callback = () => updateDimensions();
                }
                if (orientWidget) {
                    orientWidget.callback = () => updateDimensions();
                }

                // Jalankan inisialisasi awal saat node muncul di layar
                setTimeout(() => {
                    toggleVisibility();
                    checkFullBodyLock();
                    updateDimensions();
                }, 100);

                return r;
            };
        }
    }
});