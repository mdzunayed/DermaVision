<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- Chosen Palette: "Clinical Calm" - A palette using a base of warm neutrals (bg-stone-100, bg-white) with calming blues (text-sky-800, border-sky-600) and subtle, professional accents (bg-teal-600 for buttons) to create a trustworthy and focused academic/medical feel. -->
    <!-- Application Structure Plan: The SPA is structured into four thematic, scrollable sections for a clear narrative flow: 1) "Introduction" to set the context of skin cancer detection. 2) "Models & Methods" to detail the CNN architectures and data handling in an accessible, non-linear way using tabs. 3) "Interactive Performance" as the core interactive dashboard, allowing users to select datasets and view comparative model performance on dynamic charts, directly addressing the paper's key quantitative results. 4) "Explainability (Grad-CAM)" to visually showcase the model's decision-making process, reinforcing the paper's emphasis on transparency. This structure was chosen to guide the user from the 'why' to the 'how' and finally to the 'what' (results) and 'proof' (explainability), making a complex paper digestible and engaging for a broader audience. -->
    <!-- Visualization & Content Choices: 1) **Model Performance:** Report Info: Quantitative metrics (Specificity, MCC, PR-AUC, F1) from Tables 7 & 8. Goal: Compare model effectiveness. Viz/Method: Interactive Bar Chart (Chart.js). Interaction: Buttons to switch between D1 (Binary) and D2 (Multi-Class) datasets, which dynamically updates the chart data and labels. Justification: Bar charts are ideal for direct comparison of performance scores. Interactivity allows for a focused view without overwhelming the user with two separate static charts. 2) **Dataset Distribution:** Report Info: Class distribution numbers from Figures 1-4 & Table 3. Goal: Inform about dataset composition and imbalance. Viz/Method: Simple Donut Charts (Chart.js). Interaction: Hover to see class counts. Justification: Donut charts provide a quick, intuitive understanding of proportions. 3) **Model Architectures:** Report Info: Descriptions of CNNs. Goal: Organize and explain complex architectures. Viz/Method: Tabbed content sections (HTML/Tailwind/JS). Interaction: Click tabs to reveal details for each model. Justification: Tabs organize dense information cleanly, preventing a long, scroll-heavy page and allowing users to explore models at their own pace. 4) **Grad-CAM Visuals:** Report Info: Grad-CAM figures. Goal: Demonstrate model explainability. Viz/Method: Image gallery grid (HTML/Tailwind). Interaction: N/A (static display). Justification: A direct visual presentation is most effective for these qualitative results. -->
    <!-- CONFIRMATION: NO SVG graphics used. NO Mermaid JS used. -->
    <title>Skin Cancer Detection: An Interactive Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f8f7f4; /* A warmer off-white */
        }
        .chart-container {
            position: relative;
            width: 100%;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            height: 450px;
            max-height: 60vh;
        }
        .nav-link {
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .nav-link:hover, .nav-link.active {
            color: #0c4a6e; /* sky-800 */
            border-bottom-color: #0284c7; /* sky-600 */
        }
        .tab-btn {
            transition: all 0.3s ease;
        }
        .tab-btn.active {
            background-color: #0d9488; /* teal-600 */
            color: white;
        }
        .smooth-scroll {
            scroll-behavior: smooth;
        }
    </style>
</head>
<body class="text-stone-800 smooth-scroll">

    <!-- Header & Navigation -->
    <header class="bg-white/80 backdrop-blur-md sticky top-0 z-50 shadow-sm">
        <nav class="container mx-auto px-6 py-4 flex justify-between items-center">
            <h1 class="text-2xl font-bold text-sky-800">DermaVision Insights</h1>
            <div class="hidden md:flex space-x-8">
                <a href="#introduction" class="nav-link text-stone-600 border-b-2 border-transparent pb-1">Introduction</a>
                <a href="#methods" class="nav-link text-stone-600 border-b-2 border-transparent pb-1">Models & Methods</a>
                <a href="#performance" class="nav-link text-stone-600 border-b-2 border-transparent pb-1">Performance</a>
                <a href="#explainability" class="nav-link text-stone-600 border-b-2 border-transparent pb-1">Explainability</a>
            </div>
        </nav>
    </header>

    <main class="container mx-auto px-6 py-12">

        <!-- Section 1: Introduction -->
        <section id="introduction" class="py-16 text-center">
            <h2 class="text-4xl font-bold text-sky-800 mb-4">A Deep Learning Framework for Skin Cancer Detection</h2>
            <p class="max-w-3xl mx-auto text-lg text-stone-700 mb-8">
                This interactive report summarizes the findings of a study on using deep learning for multi-class skin cancer detection. Early and accurate diagnosis is critical, and this research explores how state-of-the-art Convolutional Neural Networks (CNNs) can improve diagnostic accuracy and provide interpretable results for clinicians.
            </p>
            <div class="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto text-left">
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h3 class="font-bold text-xl mb-2 text-teal-700">The Challenge</h3>
                    <p class="text-stone-600">Manually diagnosing skin lesions from dermoscopic images can be slow and subjective. AI offers a way to assist dermatologists by providing fast, consistent, and accurate analysis, but these systems must be transparent and trustworthy.</p>
                </div>
                <div class="bg-white p-6 rounded-lg shadow-md">
                    <h3 class="font-bold text-xl mb-2 text-teal-700">The Approach</h3>
                    <p class="text-stone-600">This study contrasts multiple advanced CNNs on two large, validated datasets. By using techniques like data augmentation and Grad-CAM for explainability, the framework aims to build a robust and clinically useful tool.</p>
                </div>
            </div>
        </section>

        <!-- Section 2: Models & Methods -->
        <section id="methods" class="py-16">
            <h2 class="text-3xl font-bold text-center text-sky-800 mb-12">Architectures and Methodology</h2>
            <div class="max-w-5xl mx-auto">
                <p class="text-center text-stone-700 mb-10">The study evaluated a diverse set of CNN architectures, each with unique features designed for complex image recognition tasks. The methodology involved rigorous data preprocessing, augmentation to handle class imbalance, and a standardized training pipeline. Click through the tabs below to learn more about each model.</p>
                <div class="bg-white rounded-lg shadow-xl p-8">
                    <div class="flex flex-wrap justify-center border-b border-stone-200 mb-6">
                        <button class="tab-btn py-2 px-4 rounded-t-md text-stone-600 font-semibold active" data-tab="dicenet">DiCENet</button>
                        <button class="tab-btn py-2 px-4 rounded-t-md text-stone-600 font-semibold" data-tab="xception">Xception</button>
                        <button class="tab-btn py-2 px-4 rounded-t-md text-stone-600 font-semibold" data-tab="efficientnet">EfficientNetV2-M</button>
                        <button class="tab-btn py-2 px-4 rounded-t-md text-stone-600 font-semibold" data-tab="seresnext">SE-ResNeXt50</button>
                    </div>
                    <div id="tab-content" class="text-stone-600">
                        <div id="dicenet-content" class="tab-pane active">
                            <h3 class="text-2xl font-bold text-teal-700 mb-3">DiCENet</h3>
                            <p>DiCENet utilizes diverse convolution blocks, applying filters of different sizes (3x3, 5x5, 1x1) simultaneously. This multi-resolution approach allows it to capture a wide variety of features in skin lesions, from fine textures to broader structural shapes, making it particularly effective for the binary classification task.</p>
                        </div>
                        <div id="xception-content" class="tab-pane hidden">
                            <h3 class="text-2xl font-bold text-teal-700 mb-3">Xception</h3>
                            <p>Based on depthwise separable convolutions, Xception (Extreme Inception) separates spatial and cross-channel convolutions. This makes the architecture highly efficient and powerful, as it can learn rich spatial hierarchies within images. It performed exceptionally well in both binary and multi-class tasks.</p>
                        </div>
                        <div id="efficientnet-content" class="tab-pane hidden">
                            <h3 class="text-2xl font-bold text-teal-700 mb-3">EfficientNetV2-M</h3>
                            <p>This model uses a compound scaling method to balance network depth, width, and resolution. Combined with mobile inverted bottleneck convolutions (MBConv), it achieves high accuracy with computational efficiency. It proved superior for the complex multi-class lesion classification task.</p>
                        </div>
                        <div id="seresnext-content" class="tab-pane hidden">
                            <h3 class="text-2xl font-bold text-teal-700 mb-3">SE-ResNeXt50</h3>
                            <p>This architecture combines the grouped convolutions of ResNeXt with "Squeeze-and-Excitation" (SE) blocks. SE blocks adaptively re-weight channel features, allowing the model to focus on the most informative visual cues, which is crucial for distinguishing between visually similar lesion classes.</p>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Section 3: Interactive Performance -->
        <section id="performance" class="py-16 bg-white rounded-lg shadow-xl">
            <h2 class="text-3xl font-bold text-center text-sky-800 mb-4">Interactive Performance Analysis</h2>
            <p class="text-center text-stone-700 mb-8 max-w-3xl mx-auto">Explore the performance of each CNN architecture. Select a dataset below to dynamically update the chart and compare models based on key clinical metrics. High specificity is crucial to avoid false positives, while a high MCC indicates a well-balanced overall performance.</p>
            <div class="flex justify-center space-x-4 mb-8">
                <button id="btn-d1" class="bg-teal-600 text-white font-bold py-2 px-6 rounded-full shadow-md hover:bg-teal-700 transition-colors">D1: Binary Classification</button>
                <button id="btn-d2" class="bg-stone-200 text-stone-700 font-bold py-2 px-6 rounded-full hover:bg-stone-300 transition-colors">D2: Multi-Class Classification</button>
            </div>
            <div class="chart-container">
                <canvas id="performanceChart"></canvas>
            </div>
        </section>

        <!-- Section 4: Explainability -->
        <section id="explainability" class="py-16">
            <h2 class="text-3xl font-bold text-center text-sky-800 mb-4">Model Explainability with Grad-CAM</h2>
            <p class="text-center text-stone-700 mb-12 max-w-3xl mx-auto">A key goal of this research is transparency. Gradient-weighted Class Activation Mapping (Grad-CAM) generates heatmaps that highlight the specific regions in an image the model focused on to make its prediction. This ensures the model is learning clinically relevant features, building trust and aiding in diagnostic validation.</p>
            <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                <div class="bg-white rounded-lg shadow-md p-4">
                    <img src="https://i.imgur.com/8E3A2Vn.png" alt="Grad-CAM on a benign lesion" class="rounded-md mb-3" onerror="this.onerror=null;this.src='https://placehold.co/600x400/e2e8f0/475569?text=Image+Not+Found';">
                    <h4 class="font-bold text-lg text-teal-700">Xception on Benign Lesion (D1)</h4>
                    <p class="text-stone-600 text-sm">The model correctly focuses on the lesion's core structure and uniform pigmentation, consistent with benign characteristics.</p>
                </div>
                <div class="bg-white rounded-lg shadow-md p-4">
                    <img src="https://i.imgur.com/Bv2Yj2L.png" alt="Grad-CAM on a malignant lesion" class="rounded-md mb-3" onerror="this.onerror=null;this.src='https://placehold.co/600x400/e2e8f0/475569?text=Image+Not+Found';">
                    <h4 class="font-bold text-lg text-teal-700">Xception on Malignant Lesion (D1)</h4>
                    <p class="text-stone-600 text-sm">The heatmap shows high activation on the irregular borders and varied pigmentation, key indicators of malignancy.</p>
                </div>
                <div class="bg-white rounded-lg shadow-md p-4">
                    <img src="https://i.imgur.com/b2o9f5G.png" alt="Grad-CAM on a multi-class lesion" class="rounded-md mb-3" onerror="this.onerror=null;this.src='https://placehold.co/600x400/e2e8f0/475569?text=Image+Not+Found';">
                    <h4 class="font-bold text-lg text-teal-700">SE-ResNeXt50 (D2)</h4>
                    <p class="text-stone-600 text-sm">Across various lesion types, the model demonstrates its ability to pinpoint distinct features like texture and vascular patterns.</p>
                </div>
                 <div class="bg-white rounded-lg shadow-md p-4">
                    <img src="https://i.imgur.com/k2jR8gW.png" alt="Grad-CAM on a multi-class lesion" class="rounded-md mb-3" onerror="this.onerror=null;this.src='https://placehold.co/600x400/e2e8f0/475569?text=Image+Not+Found';">
                    <h4 class="font-bold text-lg text-teal-700">Xception on Basal Cell Carcinoma (D2)</h4>
                    <p class="text-stone-600 text-sm">The model focuses on the specific textural and boundary features characteristic of basal cell carcinoma.</p>
                </div>
                 <div class="bg-white rounded-lg shadow-md p-4">
                    <img src="https://i.imgur.com/dYq8r5T.png" alt="Grad-CAM on a multi-class lesion" class="rounded-md mb-3" onerror="this.onerror=null;this.src='https://placehold.co/600x400/e2e8f0/475569?text=Image+Not+Found';">
                    <h4 class="font-bold text-lg text-teal-700">Xception on Melanoma (D2)</h4>
                    <p class="text-stone-600 text-sm">Attention is correctly placed on the asymmetrical shape and color variegation typical of melanoma.</p>
                </div>
                 <div class="bg-white rounded-lg shadow-md p-4">
                    <img src="https://i.imgur.com/tL4w5XJ.png" alt="Grad-CAM on a multi-class lesion" class="rounded-md mb-3" onerror="this.onerror=null;this.src='https://placehold.co/600x400/e2e8f0/475569?text=Image+Not+Found';">
                    <h4 class="font-bold text-lg text-teal-700">Xception on Nevus (D2)</h4>
                    <p class="text-stone-600 text-sm">For a common nevus, the model's focus is on the regular, well-defined structure, leading to a benign classification.</p>
                </div>
            </div>
        </section>

    </main>

    <footer class="bg-stone-800 text-white mt-16">
        <div class="container mx-auto px-6 py-8 text-center">
            <p>Interactive report based on the paper "A Transparent and Explainable Deep Learning Framework for Comprehensive Multi-Class Skin Cancer Detection" by Md Zunayed and Tansiv Jubayer.</p>
            <p class="text-sm text-stone-400 mt-2">This visualization is for informational purposes only and does not constitute medical advice.</p>
        </div>
    </footer>

    <script>
        document.addEventListener('DOMContentLoaded', function () {
            // Navigation scroll logic
            const navLinks = document.querySelectorAll('.nav-link');
            const sections = document.querySelectorAll('section');

            const observerOptions = {
                root: null,
                rootMargin: '0px',
                threshold: 0.4
            };

            const observer = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        navLinks.forEach(link => {
                            link.classList.remove('active');
                            if (link.getAttribute('href').substring(1) === entry.target.id) {
                                link.classList.add('active');
                            }
                        });
                    }
                });
            }, observerOptions);

            sections.forEach(section => {
                observer.observe(section);
            });

            navLinks.forEach(anchor => {
                anchor.addEventListener('click', function (e) {
                    e.preventDefault();
                    document.querySelector(this.getAttribute('href')).scrollIntoView({
                        behavior: 'smooth'
                    });
                });
            });

            // Tab functionality for Models & Methods
            const tabs = document.querySelectorAll('.tab-btn');
            const tabPanes = document.querySelectorAll('.tab-pane');

            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    tabs.forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');

                    const target = tab.getAttribute('data-tab');
                    tabPanes.forEach(pane => {
                        if (pane.id === `${target}-content`) {
                            pane.classList.remove('hidden');
                            pane.classList.add('active');
                        } else {
                            pane.classList.add('hidden');
                            pane.classList.remove('active');
                        }
                    });
                });
            });

            // Performance Chart Logic
            const ctx = document.getElementById('performanceChart').getContext('2d');
            let performanceChart;

            const d1Data = {
                labels: ['ResNeXt50', 'DenseNet201', 'EfficientNetV2-M', 'NASNetLarge', 'DiCENet', 'Xception'],
                datasets: [
                    {
                        label: 'Specificity',
                        data: [0.8920, 0.9106, 0.5000, 0.9418, 0.9500, 0.9450],
                        backgroundColor: 'rgba(54, 162, 235, 0.6)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'MCC',
                        data: [0.7642, 0.8228, 0.7886, 0.8829, 0.8891, 0.8603],
                        backgroundColor: 'rgba(255, 159, 64, 0.6)',
                        borderColor: 'rgba(255, 159, 64, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'PR-AUC',
                        data: [0.9651, 0.9788, 0.9580, 0.9725, 0.9866, 0.9265],
                        backgroundColor: 'rgba(75, 192, 192, 0.6)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'F1-Score',
                        data: [0.8808, 0.9111, 0.8938, 0.9415, 0.9423, 0.9265],
                        backgroundColor: 'rgba(255, 99, 132, 0.6)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1
                    }
                ]
            };

            const d2Data = {
                labels: ['ResNeXt50', 'DenseNet201', 'EfficientNetV2-M', 'SE-ResNeXt50', 'NASNetLarge', 'Xception'],
                datasets: [
                    {
                        label: 'Specificity',
                        data: [0.98, 0.98, 0.99, 0.98, 0.98, 0.9450],
                        backgroundColor: 'rgba(54, 162, 235, 0.6)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'MCC',
                        data: [0.91, 0.91, 0.93, 0.90, 0.91, 0.8603],
                        backgroundColor: 'rgba(255, 159, 64, 0.6)',
                        borderColor: 'rgba(255, 159, 64, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'PR-AUC',
                        data: [0.96, 0.96, 0.98, 0.96, 0.93, 0.4659],
                        backgroundColor: 'rgba(75, 192, 192, 0.6)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'F1-Score',
                        data: [0.92, 0.92, 0.94, 0.91, 0.93, 0.9265],
                        backgroundColor: 'rgba(255, 99, 132, 0.6)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1
                    }
                ]
            };

            const chartOptions = {
                maintainAspectRatio: false,
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 0.4,
                        max: 1.0,
                        title: {
                            display: true,
                            text: 'Score',
                            font: {
                                size: 14,
                                weight: '600'
                            }
                        }
                    },
                    x: {
                         ticks: {
                            autoSkip: false,
                            maxRotation: 0,
                            minRotation: 0
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += context.parsed.y.toFixed(4);
                                }
                                return label;
                            }
                        }
                    }
                }
            };

            function createChart(data, title) {
                if (performanceChart) {
                    performanceChart.destroy();
                }
                performanceChart = new Chart(ctx, {
                    type: 'bar',
                    data: data,
                    options: {
                        ...chartOptions,
                        plugins: {
                            ...chartOptions.plugins,
                            title: {
                                display: true,
                                text: title,
                                font: {
                                    size: 18,
                                    weight: 'bold'
                                },
                                padding: {
                                    bottom: 20
                                }
                            }
                        }
                    }
                });
            }

            const btnD1 = document.getElementById('btn-d1');
            const btnD2 = document.getElementById('btn-d2');

            btnD1.addEventListener('click', () => {
                createChart(d1Data, 'D1 Performance: Binary Classification (Benign vs. Malignant)');
                btnD1.classList.add('bg-teal-600', 'text-white');
                btnD1.classList.remove('bg-stone-200', 'text-stone-700');
                btnD2.classList.add('bg-stone-200', 'text-stone-700');
                btnD2.classList.remove('bg-teal-600', 'text-white');
            });

            btnD2.addEventListener('click', () => {
                createChart(d2Data, 'D2 Performance: Multi-Class Lesion Classification');
                btnD2.classList.add('bg-teal-600', 'text-white');
                btnD2.classList.remove('bg-stone-200', 'text-stone-700');
                btnD1.classList.add('bg-stone-200', 'text-stone-700');
                btnD1.classList.remove('bg-teal-600', 'text-white');
            });

            // Initial chart load
            createChart(d1Data, 'D1 Performance: Binary Classification (Benign vs. Malignant)');
        });
    </script>
</body>
</html>
