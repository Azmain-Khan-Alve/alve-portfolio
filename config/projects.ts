// src/data/project.ts
import { ValidCategory, ValidExpType, ValidSkills } from "./constants";

export interface HeroMetric {
  label: string;
  value: string;
}

export interface DescriptionSection {
  title: string;
  content: string[]; // paragraphs (supports markdown/bold)
}

export interface descriptionDetails {
  paragraphs: string[];
  bullets: string[];
}

export interface PagesInfoInterface {
  title: string;
  imgArr: string[];
  description?: string;
}

export interface TechnicalAppendix {
  title: string;
  content: string[]; // lines / paragraphs
}

export interface ProjectInterface {
  id: string;
  type: ValidExpType;
  companyName: string;
  category: ValidCategory[];
  shortDescription: string;
  websiteLink?: string;
  githubLink?: string;
  techStack: ValidSkills[];
  startDate: Date | null;
  endDate: Date | null;
  companyLogoImg: any;

  // NEW:
  heroInfo?: {
    headline?: string;
    keyMetrics?: HeroMetric[]; // e.g. [{label: 'Val Accuracy', value: '88.30%'}]
  };
  descriptionDetails?: {
    paragraphs: string[];
    bullets: string[];
  };
  
  descriptionSections?: DescriptionSection[]; // main content sections (title + paragraphs)
  bullets?: string[]; // key achievements / highlights
  pagesInfoArr: PagesInfoInterface[]; // image-driven subsections
  technicalAppendix?: TechnicalAppendix;
}

export const Projects: ProjectInterface[] = [

// ...
{
  id: "cloud-pharma",
  companyName: "Cloud Pharma",
  type: "Web/Software",
  category: ["Web Dev", "Full Stack", "E-commerce"],
  shortDescription: "End-to-end e-commerce platform for medicine sales, featuring a full admin dashboard with business insights.",
  websiteLink: undefined, // <-- Add link later if you have one
  githubLink: "https://github.com/Azmain-Khan-Alve/Cloud-Pharma", // <-- Add your GitHub link here
  techStack: ["Django", "PostgreSQL", "Bootstrap", "HTML 5", "CSS 3"],
  startDate: null, // As requested
  endDate: null,   // As requested
  companyLogoImg: "/projects/cloud-pharma-logo.png", // <-- Add a logo to /public/projects/ and change this
  
  // --- PASTE THIS NEW CASE STUDY ---
  descriptionDetails: {
    paragraphs: [
      "This project solves the dual-challenge of pharmaceutical retail: providing a seamless, secure e-commerce experience for customers while giving administrators a powerful, data-driven management dashboard. Built on a robust Django and PostgreSQL stack, it's an enterprise-ready solution."
    ],
    bullets: [
      "Full E-Commerce Workflow: Secure user authentication, product catalog, cart, and checkout.",
      "Powerful Admin Dashboard: Full CRUD control over products, orders, and customer data.",
      "Business Insights Engine: Integrated statistics module to track sales, customer growth, and order volume.",
      "Built for Scale: Architected with a clean, logical separation for future enhancements."
    ],
  },
  pagesInfoArr: [
    {
      title: "Secure User Authentication",
      description: "A secure and straightforward registration and login system is the first step in building customer trust.",
      imgArr: [
        "/projects/cloud-pharma/register.png",
        "/projects/cloud-pharma/login.png"
      ]
    },
    {
      title: "Intuitive E-commerce Workflow",
      description: "A streamlined user journey, from searching for products to completing a multi-step checkout.",
      imgArr: [
        "/projects/cloud-pharma/store.png",
        "/projects/cloud-pharma/medicine.png",
        "/projects/cloud-pharma/cart.png",
        "/projects/cloud-pharma/checkout.png"
      ]
    }
  ]
  // --- END OF NEW CASE STUDY ---
},
// ...
  // ===============================================================================
// 🔹 Efficient & Explainable Skin Lesion Classification
// ===============================================================================
{
  id: "skin-lesion-classification",
  companyName: "Efficient & Explainable Skin Lesion Classification",
  type: "AI/ML",
  category: ["AI/ML", "Deep Learning", "Healthcare"],
  shortDescription:
    "A single ConvNeXt-Large model that matches ensemble performance on clinical metrics, built for safety and interpretability.",
  websiteLink: undefined,
  githubLink: "https://github.com/Azmain-Khan-Alve/Diabetes-Risk-Prediction", // <-- This was in your previous version
  techStack: ["PyTorch", "TensorFlow", "Keras", "Scikit-Learn", "Hugging Face"], // <-- From your previous version
  startDate: new Date("2024-05-01"), // <-- From your previous version
  endDate: new Date("2025-05-01"), // <-- From your previous version
  companyLogoImg: "/projects/skin-lesion-logo.png", // <-- From your previous version

  // 🔹 Hero metrics to show at top
  heroInfo: {
    headline:
      "Early detection of malignant skin lesions saves lives, but automated dermoscopic classification is challenged by severe class imbalance, subtle inter-class visual differences, and image artifacts. This project demonstrates that — with careful data engineering and an optimized training pipeline — a single ConvNeXt-Large model can match or surpass heavier ensemble solutions on metrics that matter clinically (sensitivity & specificity), while also providing visual explanations through Grad-CAM. The result is a practical, deployable classifier that emphasizes diagnostic safety and interpretability.",
    keyMetrics: [
      { label: "Validation Accuracy", value: "88.30%" },
      { label: "Macro Sensitivity", value: "86.27%" },
      { label: "Specificity", value: "98.08%" },
      { label: "Melanoma Recall", value: "≈87.7%" },
    ],
  },

  // 🔹 Main narrative sections
  descriptionSections: [
    {
      title: "The Problem & Clinical Motivation",
      content: [
        "Dermoscopic images vary widely across lesion type, skin tone, and image capture conditions. Benchmarks such as ISIC-2019 contain eight lesion classes but suffer extreme imbalance (Nevus dominates), which can hide poor detection of life-threatening classes like Melanoma and SCC. This leads to models that look accurate overall but risk missing rare malignant cases — a critical failure mode in healthcare. The project tackles this by optimizing for balanced clinical performance, not just overall accuracy.",
      ],
    },
    {
      title: "Hypothesis & Design Principle",
      content: [
        "Rather than relying on expensive ensembles, we hypothesized that a single modern ConvNet (ConvNeXt-Large pre-trained on ImageNet-22K) plus a carefully engineered training pipeline (targeted augmentation, focal loss with class weighting, staged fine-tuning) would achieve robust, class-balanced performance and be easier to deploy. We also required interpretability — every prediction must be explainable via Grad-CAM to build clinical trust.",
      ],
    },
    {
      title: "What I Built (Approach)",
      content: [
        "• Consolidated ISIC training + labeled test images into a larger dataset (≈31,522 images) and used stratified splits for train/val/test.",
        "• Balanced classes by augmenting minority classes until each had ≈3,500 training images using a medically realistic Albumentations pipeline (flips, rotations, color jitter, coarse dropout, blur, elastic transforms).",
        "• Fine-tuned `convnext_large.fb_in22k` with AdamW + CosineAnnealingWarmRestarts, Focal Loss (γ=4.0, label smoothing=0.08), and class weights boosted for high-risk classes (Melanoma ×4.2, SCC ×3.8, AK ×5.0).",
        "• Integrated Grad-CAM applied to the final ConvNeXt stage for per-image visual explanations.",
      ],
    },
    {
      title: "Training & Validation Protocol",
      content: [
        "Full retraining from scratch (0 → 130 epochs) confirmed the stability of the optimized pipeline and reproduced checkpoint best results. Mixed precision training, accumulated gradients, stochastic depth, and label smoothing were used to improve stability and generalization.",
      ],
    },
    {
      title: "Key Results",
      content: [
        "• **Validation accuracy:** 88.30% — **Test accuracy:** 87.97%.",
        "• **Macro sensitivity:** 86.27% (val) / 86.31% (test).",
        "• **Estimated specificity (val):** 98.08%.",
        "• **Per-class highlights:** Melanoma recall ≈ 87.7%; BCC recall ≈ 96.8%; minority classes (VASC, DF) show high precision/recall after augmentation. These balanced metrics demonstrate clinically safer behavior than prior ConvNeXt ensembles.",
      ],
    },
    {
      title: "Explainability & Clinical Safety",
      content: [
        "Grad-CAM heatmaps reliably focus on lesion regions (edges, pigment variation, structural irregularities) rather than background artifacts, supporting clinical verification and error analysis. Misclassifications tend to be conservative (overcalling benign as malignant), which is preferable clinically because it reduces missed malignancies.",
      ],
    },
    {
      title: "Why This Matters (Impact Statement)",
      content: [
        "This work shows a single, production-feasible model can be: (a) more clinically reliable in detecting dangerous lesions than larger ensembles, (b) far less expensive to serve, and (c) interpretable for clinician review — making the model practical for telemedicine, triage systems, or integration in dermatology workflows.",
      ],
    },
  ],

  // 🔹 Key achievements (summary bullets)
  bullets: [
    "Achieved **88.30% validation accuracy** and **87.97% test accuracy** with a single ConvNeXt-Large model.",
    "Macro sensitivity **86.27%** (validation) — clinically important classes (Melanoma, BCC) were detected with high recall (Melanoma ≈87.7%, BCC ≈96.8%).",
    "Specificity **≈98.08%** on validation, reducing false alarms while maintaining high sensitivity.",
    "Integrated **Grad-CAM** for per-image heatmaps that align with dermatological features (asymmetry, border, color variation) to increase clinician trust.",
    "Demonstrated that a single optimized ConvNeXt-Large **outperforms ConvNeXt ensembles** on sensitivity and specificity while being simpler and cheaper to deploy.",
  ],

  // 🔹 Individual case study sections (for image & subtopics)
  pagesInfoArr: [
    {
      title: "The Challenge — Dataset Imbalance & Visual Ambiguity",
      description:
        "ISIC-2019 contains 31k consolidated images across 8 classes but is heavily skewed (Nevus dominant). We expanded the dataset, removed unknown labels, and analyzed per-class distribution to drive augmentation strategy.",
      imgArr: [
        "/projects/skin-lesion/pie chart.png",
        "/projects/skin-lesion/sample_grid.png",
      ],
    },
    {
      title: "Training Pipeline — Targeted Augmentation & Loss Engineering",
      description:
        "We balanced classes to ~3.5k images/class using Albumentations and used Focal Loss + manual class weights (Melanoma ×4.2, SCC ×3.8, AK ×5.0) to prioritize high-risk classes.",
      imgArr: ["/projects/skin-lesion/augmentation_examples.png"],
    },
    {
      title: "Model & Performance — ConvNeXt-Large vs Ensembles",
      description:
        "A single ConvNeXt-Large model (pretrained on ImageNet-22K) achieves better sensitivity & specificity than reported ConvNeXt ensembles, with significantly lower parameter and deployment cost.",
      imgArr: ["/projects/skin-lesion/comparison.png"],
    },
    {
      title: "Explainability — Grad-CAM Visualizations",
      description:
        "Grad-CAM heatmaps validate model focus on clinically relevant lesion regions and highlight conservative error patterns (overcalling ambiguous lesions). Useful for clinician review & triage.",
      imgArr: [
        "/projects/skin-lesion/GRAD CAM.png",
        "/projects/skin-lesion/gradcam_grid.png",
      ],
    },
    {
      title: "Technical Appendix / Reproducibility",
      description:
        "Hyperparameters, augmentation pipeline, class weighting formula, training schedule, and links to checkpoints & scripts (or GitHub). — (expandable panel).",
      imgArr: [],
    },
  ],

  // 🔹 Technical appendix (optional collapsible section)
  technicalAppendix: {
    title: "Technical Details — Full Training Specification",
    content: [
      "**Dataset & Splits:**",
      "Consolidated dataset after merging labeled test images → 31,522 images. Stratified split: train 70% / val 20% / test 10%.",
      "",
      "**Augmentation Pipeline (Albumentations highlights):**",
      "• **Flip:** Horizontal/Vertical, RandomRotate90",
      "• **Geometric:** ShiftScaleRotate, RandomGridShuffle (4×4)",
      "• **Photometric:** ColorJitter (brightness/contrast/saturation=0.2, p=0.7)",
      "• **Noise/Blur:** GaussianNoise, GaussianBlur (limit 3–5 px)",
      "• **Occlusion:** CoarseDropout (3–8 holes, 16–48 px)",
      "• **Elastic/Distort:** Optical distortion, ElasticTransform",
      "• **Input resize:** 224×224, normalized with ImageNet mean/std.",
      "",
      "**Model & Training:**",
      "• **Architecture:** `convnext_large.fb_in22k` (pretrained) → GlobalAvgPool → Linear(8) + Softmax.",
      "• **Optimizer:** AdamW, lr base = 5e-6, weight_decay=1e-5, accumulated gradients.",
      "• **Scheduler:** CosineAnnealingWarmRestarts",
      "• **Loss:** FocalLoss (γ=4.0) + label smoothing (0.08) + manual class weights (boosted: Melanoma×4.2, SCC×3.8, AK×5.0).",
      "• **Training:** Full retrain 0→130 epochs, mixed precision, stochastic depth.",
      "",
      "**Evaluation:**",
      "• **Metrics:** Accuracy, Sensitivity (Recall), Specificity, Precision, F1 (macro & weighted).",
      "• **Final:** Val acc 88.30%, macro sensitivity 86.27%, specificity 98.08% (val); test metrics stable and similar.",
    ],
  },
},
  // inside @/config/projects (replace the skin-lesion object)
  // {
  //   id: "skin-lesion-classification",
  //   companyName: "Efficient & Explainable Skin Lesion Classification",
  //   type: "AI/ML",
  //   category: ["AI/ML", "Deep Learning", "Healthcare"],
  //   shortDescription:
  //     "Single ConvNeXt-Large model that surpasses ensemble baselines in clinically relevant metrics while remaining lightweight and interpretable (Grad-CAM).",
  //   websiteLink: undefined,
  //   githubLink: "https://github.com/Azmain-Khan-Alve/Diabetes-Risk-Prediction",
  //   techStack: ["PyTorch", "TensorFlow", "Keras", "Scikit-Learn", "Hugging Face"],
  //   startDate: new Date("2024-05-01"),
  //   endDate: new Date("2025-05-01"),
  //   companyLogoImg: "/projects/skin-lesion-logo.png",

  //   // NEW OPTIONAL FIELDS for richer UI
  //   heroInfo: {
  //     headline:
  //       "Efficient & Explainable Skin Lesion Classification — Single ConvNeXt-Large with clinical-grade metrics",
  //     keyMetrics: [
  //       { label: "Validation Accuracy", value: "88.30%" },
  //       { label: "Macro Sensitivity", value: "86.27%" },
  //       { label: "Specificity (val)", value: "98.08%" }
  //     ]
  //   },

  //   // NEW: sections with titles
  //   descriptionSections: [
  //     {
  //       title: "Project Summary",
  //       content: [
  //         "Early detection of malignant skin lesions saves lives. This project demonstrates that with careful data engineering and an optimized training pipeline, a single ConvNeXt-Large model can match or surpass heavy ensemble solutions while remaining interpretable using Grad-CAM."
  //       ]
  //     },
  //     {
  //       title: "Problem & Clinical Motivation",
  //       content: [
  //         "Benchmarks like ISIC-2019 are heavily imbalanced (Nevus dominates). Models that show high overall accuracy often fail on rare but dangerous classes like Melanoma or SCC. This project optimizes for balanced clinical performance rather than only overall accuracy."
  //       ]
  //     },
  //     {
  //       title: "Hypothesis & Design",
  //       content: [
  //         "Hypothesis: a single ConvNeXt-Large (pretrained on ImageNet-22K) combined with targeted augmentation, focal loss + class weighting, and staged fine-tuning can outperform ensembles while being easier to deploy. Interpretability via Grad-CAM is required for clinical trust."
  //       ]
  //     },
  //     {
  //       title: "Approach (What I built)",
  //       content: [
  //         "• Consolidated ISIC training + labeled test images into ~31,522 images with stratified splits.",
  //         "• Balanced minority classes to ~3.5k samples with Albumentations (flips, rotations, color jitter, coarse dropout, blur, elastic transforms).",
  //         "• Fine-tuned convnext_large.fb_in22k using AdamW + CosineAnnealingWarmRestarts; used Focal Loss (γ=4.0) with label smoothing (0.08) and class weight boosts for Melanoma×4.2, SCC×3.8, AK×5.0.",
  //         "• Integrated Grad-CAM for per-image visual explanations."
  //       ]
  //     },
  //     {
  //       title: "Training & Validation",
  //       content: [
  //         "Full retraining (0→130 epochs) confirmed stability. Used mixed precision, gradient accumulation, stochastic depth, and label smoothing for better generalization."
  //       ]
  //     },
  //     {
  //       title: "Impact & Deployment Potential",
  //       content: [
  //         "A single production-feasible model that outperforms ensembles on clinically-relevant metrics while being cheaper and faster to deploy — suitable for telemedicine triage and hospital workflows."
  //       ]
  //     }
  //   ],

  //   // Keep your old structure too (for backward compatibility)
  //   descriptionDetails: {
  //     paragraphs: [
  //       "**The Problem:** Skin cancer is a critical health issue where early detection is paramount. However, visual diagnosis is challenging. Malignant lesions often look identical to benign nevi, and datasets like ISIC-2019 are difficult due to class imbalance and artifacts.",
  //       "**The Goal & Hypothesis:** Most SOTA use heavy ensembles. I hypothesised that a single, well-trained model (ConvNeXt-Large) could match or surpass them with a tailored pipeline.",
  //       "**My Process:** I engineered the training pipeline, used targeted augmentation, class-weighted focal loss, and Grad-CAM for explainability."
  //     ],
  //     bullets: [
  //       "Achieved **88.30% validation accuracy** and **87.97% test accuracy** with a single model.",
  //       "**Macro sensitivity 86.27%** (validation).",
  //       "Integrated **Grad-CAM** for transparent explanations."
  //     ]
  //   },

  //   pagesInfoArr: [
  //     {
  //       title: "The Challenge — Dataset Imbalance",
  //       description:
  //         "ISIC-2019 contains ~31k images across 8 classes but is highly skewed. Targeted augmentation and class weighting were used to fix the imbalance.",
  //       imgArr: ["/projects/skin-lesion/pie-chart.png", "/projects/skin-lesion/sample-grid.png"]
  //     },
  //     {
  //       title: "Training Pipeline — Augmentation & Loss",
  //       description:
  //         "Minority classes were augmented using Albumentations and training used focal loss with class boosts for high-risk categories.",
  //       imgArr: ["/projects/skin-lesion/augmentation-examples.png"]
  //     },
  //     {
  //       title: "Explainability — Grad-CAM",
  //       description:
  //         "Grad-CAM visualizations validate model attention aligns with lesion regions (pigmentation, borders, asymmetry).",
  //       imgArr: ["/projects/skin-lesion/grad-cam.png", "/projects/skin-lesion/gradcam-grid.png"]
  //     },
  //     {
  //       title: "Performance — Single Model vs Ensembles",
  //       description:
  //         "A single ConvNeXt-Large achieved higher clinical sensitivity & specificity than ensemble baselines while being cheaper to deploy.",
  //       imgArr: ["/projects/skin-lesion/comparison.png"]
  //     }
  //   ],

  //   technicalAppendix: {
  //     title: "Technical Appendix",
  //     content: [
  //       "Dataset: Consolidated ~31,522 images; stratified splits: train 70% / val 20% / test 10%.",
  //       "Augmentations: Flip, Rotate, ShiftScaleRotate, ColorJitter, CoarseDropout, GaussianBlur, ElasticTransform. Resize to 224×224 and ImageNet normalization.",
  //       "Model: convnext_large.fb_in22k -> GlobalAvgPool -> Linear(8).",
  //       "Optimizer: AdamW (lr ~5e-6), CosineAnnealingWarmRestarts, mixed precision.",
  //       "Loss: FocalLoss (γ=4.0) + label smoothing 0.08 + manual class weights with boosts for Melanoma/SCC/AK.",
  //       "Training: 130 epochs, early stopping on validation F1/sensitivity."
  //     ]
  //   }
  // },
// ===============================================================================
// 🔹 Add this object to your Projects array in src/data/project.ts
// ===============================================================================

// ===============================================================================
// 🔹 Diabetes Risk Prediction
// ===============================================================================
{
  id: "diabetes-risk-prediction",
  companyName: "Diabetes Risk Prediction using Machine Learning",
  type: "AI/ML",
  category: ["AI/ML", "Machine Learning", "Healthcare"],
  shortDescription:
    "Built an end-to-end ML pipeline that predicts diabetes onset, achieving 93% accuracy and 0.96 ROC-AUC with Random Forest.",
  websiteLink: undefined,
  githubLink: "https://github.com/Azmain-Khan-Alve/Diabetes-Risk-Prediction",
  techStack: ["Python", "Scikit-Learn", "Pandas", "Matplotlib", "Seaborn"],
  startDate: null, // <-- IMPORTANT: Change this date
  endDate: null, // <-- IMPORTANT: Change this date
  companyLogoImg: "/projects/diabetes/diabetes-logo.png", // <-- IMPORTANT: Add a logo and change this path

  // 🔹 Hero metrics to show at top
  heroInfo: {
    headline:
      "Built an interpretable, clinically aligned ML model for diabetes risk prediction — achieving 93% accuracy with Random Forest.",
    keyMetrics: [
      { label: "Accuracy", value: "93%" },
      { label: "ROC-AUC", value: "0.96" },
      { label: "Top Model", value: "Random Forest" },
    ],
  },

  // 🔹 Main narrative sections
  descriptionSections: [
    {
      title: "Project Summary",
      content: [
        "Early detection of diabetes enables timely lifestyle and clinical intervention, yet many at-risk individuals remain undiagnosed due to costly or invasive tests. This project explores how traditional ML techniques can provide a low-cost, data-driven pre-screening solution using basic medical attributes (e.g., glucose level, BMI, age, smoking status, family history).",
        "The pipeline handles data imbalance, performs careful feature standardization, and compares several algorithms to determine the optimal model for balanced accuracy and interpretability.",
      ],
    },
    {
      title: "Problem & Motivation",
      content: [
        "The World Health Organization identifies diabetes as one of the fastest-growing global health challenges. Existing clinical methods for diagnosis rely on laboratory tests (e.g., fasting plasma glucose), which are not always accessible or cost-efficient.",
        "**Our goal:** Build a predictive system that can estimate diabetes risk with minimal inputs — enabling large-scale community screening and healthcare triage.",
      ],
    },
    {
      title: "Hypothesis & Approach",
      content: [
        "We hypothesized that traditional supervised ML algorithms — when trained on a cleaned, scaled dataset — can match or exceed the performance of more complex models for tabular medical data.",
        "The pipeline was designed to:",
        "• **Standardize features** for scale-sensitive algorithms (SVM, KNN).",
        "• **Benchmark multiple classifiers** to compare bias-variance trade-offs.",
        "• **Evaluate with clinically relevant metrics** (Precision, Recall, F1-score, ROC-AUC) to ensure both accuracy and safety.",
      ],
    },
    {
      title: "1. Data Preparation",
      content: [
        "Loaded `diabetes_prediction_dataset.csv` and verified shape, null values, and feature types. The dataset was clean with no missing values.",
        "Features included: Age, Gender, Hypertension, Heart Disease, BMI, Smoking History, HbA1c level, Blood Glucose level, and Outcome.",
        "Applied `StandardScaler` to normalize continuous features (e.g., BMI, glucose, HbA1c).",
        "Used `train_test_split` (80/20) with stratification to preserve outcome distribution.",
      ],
    },
    {
      title: "2. Model Training & Evaluation",
      content: [
        "Benchmarked 5 classical ML models: K-Nearest Neighbors (KNN), Logistic Regression, Support Vector Machine (SVM), Decision Tree, and Random Forest.",
        "All models were trained using cross-validation (StratifiedKFold) and tuned with `GridSearchCV` for hyperparameter optimization.",
      ],
    },
    {
      title: "3. Key Metrics & Results",
      content: [
        "**Random Forest** emerged as the most balanced model, combining high sensitivity (detecting diabetic cases) with strong specificity (avoiding false positives).",
        "It achieved **93% Accuracy** and a **ROC-AUC of 0.96**, significantly outperforming the other benchmarked models like Logistic Regression (0.92 ROC-AUC) and SVM (0.94 ROC-AUC).",
        // The table from your notes is better visualized on the frontend,
        // but this summary sentence captures the key result.
      ],
    },
    {
      title: "4. Model Explainability & Visualization",
      content: [
        "Plotted confusion matrices to visualize class-wise performance and ROC curves to confirm robust class separation.",
        "Generated a feature importance plot, which revealed that **Blood Glucose, HbA1c, and BMI** were the top predictors, aligning with established medical literature.",
        "Used Seaborn heatmaps for correlation analysis and Plotly for interactive EDA visualization.",
      ],
    },
    {
      title: "Impact & Insights",
      content: [
        "This project demonstrates how simple yet optimized ML models can:",
        "• Detect high-risk individuals early with over 90% accuracy.",
        "• Offer an interpretable, cost-effective alternative to deep learning models.",
        "• Be deployed in low-resource settings for screening or telemedicine.",
        "From a healthcare standpoint, the Random Forest model’s high recall ensures minimal missed diagnoses — crucial in preventive medicine.",
      ],
    },
  ],

  // 🔹 Key achievements (summary bullets)
  bullets: [
    "Built an end-to-end ML pipeline — from data preprocessing to evaluation and visualization.",
    "Benchmarked 5 algorithms; achieved 93% accuracy and ROC-AUC 0.96 with Random Forest.",
    "Implemented cross-validation & hyperparameter tuning (GridSearchCV) for generalization.",
    "Generated feature importance and explainability visuals for medical interpretability.",
    "Delivered a deployable model ready for integration into a Flask or Gradio web app.",
  ],

  // 🔹 Individual case study sections (for image & subtopics)
  pagesInfoArr: [
    {
      title: "EDA & Dataset Overview",
      description:
        "Showed dataset summary, feature correlation, and class imbalance visualization.",
      imgArr: [
        "/projects/diabetes/eda.png",
        "/projects/diabetes/correlation_heatmap.png",
      ],
    },
    {
      title: "Model Benchmarking",
      description:
        "Compared model metrics; showed accuracy, precision, recall, and ROC curves.",
      imgArr: [
        "/projects/diabetes/roc_curve.png",
        "/projects/diabetes/confusion_matrix.png",
      ],
    },
    {
      title: "Feature Importance & Explainability",
      description:
        "Highlighted medically relevant features (Glucose, HbA1c, BMI).",
      imgArr: ["/projects/diabetes/feature_importance.png"],
    },
    {
      title: "Final Model — Random Forest",
      description: "Showed results summary and real-world implications.",
      imgArr: ["/projects/diabetes/results_summary.png"],
    },
  ],

  // 🔹 Technical appendix (optional collapsible section)
  technicalAppendix: {
    title: "Technical Appendix — Reproducibility",
    content: [
      "Dataset: 100,000 samples × 9 features",
      "Target: Binary (0 = non-diabetic, 1 = diabetic)",
      "Split: Stratified 80/20 train-test split",
      "Preprocessing: StandardScaler normalization, Label encoding",
      "Model: RandomForestClassifier",
      "Hyperparameters: n_estimators=200, max_depth=12, min_samples_split=4, class_weight='balanced'",
      "Tuning: GridSearchCV (5-fold CV)",
      "Evaluation: Accuracy, F1, Precision, Recall, ROC-AUC",
      "Key Results: Validation Accuracy (93%), ROC-AUC (0.96)",
    ],
  },
},

// ===============================================================================
  
// {
//   id: "cholo-pothik",
//   companyName: "Cholo Pothik",
//   type: "Web/Software",
//   category: ["Web Dev", "Full Stack", "Travel"],
//   shortDescription: "Full-stack Travel Management System with separate user and admin interfaces.",
//   websiteLink: undefined, // <-- Add link later if you have one
//   githubLink: "https://github.com/Azmain-Khan-Alve/Cholo-Pothik", 
//   techStack: ["PHP", "MySQL", "HTML 5", "CSS 3"],
//   startDate: null, // <-- IMPORTANT: Change this date
//   endDate: null, // <-- IMPORTANT: Change this date
//   companyLogoImg: "/projects/placeholder.png", // <-- IMPORTANT: Add a logo to /public/projects/ and change this path
  
//   pagesInfoArr: [], 
//   descriptionDetails: {
//     paragraphs: [
//       "A full-stack Travel Management System built from scratch, featuring separate, secure interfaces for both users and administrators."
//     ],
//     bullets: [
//       "User Interface: Features user authentication, destination search, hotel booking capabilities, and a feedback system.",
//       "Admin Interface: Provides full Create, Read, Update, and Delete (CRUD) operations for managing places, hotels, and travel agents."
//     ],
//   },
// },

// ===============================================================================


// {
//   id: "skin-lesion-classification",
//   companyName: "Efficient & Explainable Skin Lesion Classification",
//   type: "AI/ML",
//   category: ["AI/ML", "Deep Learning", "Healthcare"],
//   shortDescription:
//     "Single ConvNeXt-Large model that surpasses ConvNeXt ensembles on clinically relevant metrics while remaining lightweight and interpretable through Grad-CAM.",
//   websiteLink: undefined,
//   githubLink: "https://github.com/Azmain-Khan-Alve/Diabetes-Risk-Prediction",
//   techStack: ["PyTorch", "TensorFlow", "Keras", "Scikit-Learn", "Hugging Face"],
//   startDate: new Date("2024-05-01"),
//   endDate: new Date("2025-05-01"),
//   companyLogoImg: "/projects/skin-lesion-logo.png",

//   // 🔹 Hero metrics to show at top
//   heroInfo: {
//     headline:
//       "Single ConvNeXt-Large model achieving state-of-the-art results on ISIC-2019 while remaining interpretable and deployable.",
//     keyMetrics: [
//       { label: "Validation Accuracy", value: "88.30%" },
//       { label: "Macro Sensitivity", value: "86.27%" },
//       { label: "Specificity", value: "98.08%" },
//     ],
//   },

//   // 🔹 Main narrative sections
//   descriptionSections: [
//     {
//       title: "Project Summary",
//       content: [
//         "Early detection of malignant skin lesions saves lives, but automating this task with AI is difficult due to class imbalance, subtle inter-class variations, and visual artifacts. In this project, I built a single ConvNeXt-Large model that rivals and even surpasses complex ensemble systems. With careful data engineering, augmentation, and a fine-tuned training pipeline, the model achieved clinical-grade performance while remaining interpretable through Grad-CAM.",
//       ],
//     },
//     {
//       title: "The Problem & Clinical Motivation",
//       content: [
//         "Dermoscopic images vary drastically across lesion types, patient skin tones, and lighting conditions. The ISIC-2019 benchmark includes eight classes but suffers extreme imbalance—benign classes dominate, while dangerous ones like Melanoma or SCC are underrepresented. This can lead models to appear accurate while still missing rare malignancies. The goal here was to achieve balanced, clinically safe detection rather than just high overall accuracy.",
//       ],
//     },
//     {
//       title: "Hypothesis & Design Principle",
//       content: [
//         "Most existing SOTA systems rely on large model ensembles that are accurate but too heavy for real-world use. My hypothesis was that a single, modern ConvNeXt-Large model (pretrained on ImageNet-22K), trained with a carefully engineered pipeline—targeted augmentation, focal loss, and adaptive class weighting—could outperform these ensembles while being efficient and interpretable.",
//       ],
//     },
//     {
//       title: "What I Built (Approach)",
//       content: [
//         "• Consolidated the ISIC-2019 dataset and merged labeled test images, creating a 31,522-image dataset split by stratified sampling for training, validation, and testing.",
//         "• Applied advanced augmentations to balance each class (~3.5k images per class) using Albumentations: flips, rotations, color jitter, elastic distortions, and coarse dropout.",
//         "• Fine-tuned a ConvNeXt-Large model using AdamW optimizer, Focal Loss (γ=4.0), label smoothing (0.08), and manually boosted class weights for high-risk categories (Melanoma ×4.2, SCC ×3.8, AK ×5.0).",
//         "• Integrated Grad-CAM at the final ConvNeXt stage to visualize and verify that the model’s focus aligned with clinically relevant lesion regions.",
//       ],
//     },
//     {
//       title: "Training & Validation",
//       content: [
//         "The model was retrained from scratch for 130 epochs using mixed-precision training, gradient accumulation, stochastic depth, and label smoothing for stability. The optimized pipeline consistently reproduced best checkpoints with strong generalization.",
//       ],
//     },
//     {
//       title: "Key Results",
//       content: [
//         "Validation accuracy: 88.30%, Test accuracy: 87.97%.",
//         "Macro sensitivity: 86.27% (validation) / 86.31% (test).",
//         "Estimated specificity: 98.08% (validation).",
//         "Class-wise: Melanoma recall ≈ 87.7%, BCC recall ≈ 96.8%. Minority classes (VASC, DF) achieved strong precision and recall after augmentation. The single model showed more balanced and clinically safer behavior than ensemble approaches.",
//       ],
//     },
//     {
//       title: "Explainability & Clinical Safety",
//       content: [
//         "Grad-CAM heatmaps consistently highlighted meaningful lesion structures—edges, pigment irregularities, and texture patterns—rather than background noise. The model’s misclassifications were conservative: it tended to over-diagnose benign lesions as malignant rather than miss dangerous cases, which is safer in clinical settings.",
//       ],
//     },
//     {
//       title: "Impact & Real-World Value",
//       content: [
//         "This project demonstrates that one well-optimized, explainable deep learning model can outperform complex ensembles. It’s lightweight, efficient, and trustworthy—making it a viable candidate for telemedicine platforms, triage tools, or dermatology support systems.",
//       ],
//     },
//   ],

//   // 🔹 Key achievements (summary bullets)
//   bullets: [
//     "Achieved 88.30% validation and 87.97% test accuracy with a single ConvNeXt-Large model.",
//     "Reached macro sensitivity of 86.27% and specificity of 98.08%, ensuring balanced, clinically reliable performance.",
//     "Outperformed reported ConvNeXt ensembles while reducing computational cost and deployment complexity.",
//     "Integrated Grad-CAM for per-image heatmaps aligning with dermatological features like asymmetry, border, and pigment variation.",
//     "Built a model prioritizing interpretability and diagnostic safety — suitable for telemedicine and hospital workflows.",
//   ],

//   // 🔹 Individual case study sections (for image & subtopics)
//   pagesInfoArr: [
//     {
//       title: "The Challenge — Dataset Imbalance & Visual Ambiguity",
//       description:
//         "ISIC-2019’s dataset (31k+ images across 8 lesion classes) is heavily skewed toward benign Nevus samples. We addressed this imbalance through augmentation and weighted losses, ensuring rare malignant cases were properly learned.",
//       imgArr: ["/projects/skin-lesion/pie chart.png", "/projects/skin-lesion/sample_grid.png"],
//     },
//     {
//       title: "Training Pipeline — Augmentation & Loss Engineering",
//       description:
//         "Balanced classes to ~3.5k images each using Albumentations (rotation, jitter, dropout, blur). Focal Loss with adaptive weighting (Melanoma ×4.2, SCC ×3.8, AK ×5.0) helped the model focus on underrepresented, high-risk classes.",
//       imgArr: ["/projects/skin-lesion/augmentation_examples.png"],
//     },
//     {
//       title: "Model & Performance — ConvNeXt-Large vs Ensembles",
//       description:
//         "Our single ConvNeXt-Large model outperformed ConvNeXt ensemble baselines on sensitivity and specificity, with much lower computational cost and deployment footprint.",
//       imgArr: ["/projects/skin-lesion/comparison.png"],
//     },
//     {
//       title: "Explainability — Grad-CAM Visualizations",
//       description:
//         "Grad-CAM visualizations confirm that the model’s attention focuses on lesion regions rather than background noise. These insights build clinical trust and improve post-hoc error analysis.",
//       imgArr: ["/projects/skin-lesion/GRAD CAM.png", "/projects/skin-lesion/gradcam_grid.png"],
//     },
//   ],

//   // 🔹 Technical appendix (optional collapsible section)
//   technicalAppendix: {
//     title: "Technical Appendix — Reproducibility",
//     content: [
//       "Dataset: 31,522 dermoscopic images (train/val/test = 70/20/10).",
//       "Input size: 224×224, normalized using ImageNet mean/std.",
//       "Augmentations: flips, rotations, color jitter, Gaussian blur, CoarseDropout, ElasticTransform.",
//       "Optimizer: AdamW (lr = 5e-6, weight_decay = 1e-5). Scheduler: CosineAnnealingWarmRestarts.",
//       "Loss: FocalLoss (γ=4.0) with label smoothing (0.08) and boosted class weights (Melanoma×4.2, SCC×3.8, AK×5.0).",
//       "Training: 130 epochs, mixed precision, stochastic depth, gradient accumulation.",
//       "Evaluation metrics: Accuracy, Sensitivity, Specificity, Precision, F1 (macro & weighted).",
//       "Final metrics — Val: 88.30% acc, 86.27% sensitivity, 98.08% specificity; Test: 87.97% acc, 86.31% sensitivity.",
//     ],
//   },
// },

// ====================================================================================
{
  id: "skin-lesion-classification",
  companyName: "Efficient & Explainable Skin Lesion Classification",
  type: "AI/ML",
  category: ["AI/ML", "Deep Learning", "Healthcare"],
  shortDescription:
    "Single ConvNeXt-Large model that surpasses ConvNeXt ensembles on clinically relevant metrics while remaining lightweight and interpretable using Grad-CAM.",
  websiteLink: undefined,
  githubLink: "https://github.com/Azmain-Khan-Alve/Diabetes-Risk-Prediction",
  techStack: ["PyTorch", "TensorFlow", "Keras", "Scikit-Learn", "Hugging Face"],
  startDate: new Date("2024-05-01"),
  endDate: new Date("2025-05-01"),
  companyLogoImg: "/projects/skin-lesion-logo.png",

  // --- DETAILED DESCRIPTION ---
  descriptionDetails: {
    paragraphs: [
      "**Project Summary:** Early detection of malignant skin lesions saves lives, but automated dermoscopic classification is extremely challenging. The task is complicated by severe class imbalance, subtle visual differences between lesion types, and artifacts such as hairs or ruler marks. This project demonstrates that, with careful data engineering and an optimized training pipeline, a single ConvNeXt-Large model can match or surpass heavier ensemble architectures on key clinical metrics (sensitivity and specificity). Through Grad-CAM visualization, the model also provides interpretable insights, making it both high-performing and clinically trustworthy.",

      "**The Problem & Clinical Motivation:** Dermoscopic images show large variations in color, shape, and lighting. The ISIC-2019 dataset includes eight lesion classes but is heavily imbalanced—benign nevi dominate, while malignant lesions like melanoma or SCC are rare. Many models trained naively on this dataset appear accurate overall but perform poorly on the critical minority classes, risking dangerous false negatives. The aim of this project was to build a model that achieves balanced, clinically safe results rather than simply maximizing raw accuracy.",

      "**Hypothesis & Design Principle:** Instead of relying on expensive ensembles, I hypothesized that a single modern ConvNet—ConvNeXt-Large pretrained on ImageNet-22K—combined with an optimized training pipeline (targeted data augmentation, class-weighted Focal Loss, and staged fine-tuning) could reach or exceed ensemble performance while being easier to deploy. Another core design goal was interpretability: every prediction should be explainable through Grad-CAM to foster clinical confidence.",

      "**What I Built (Approach):** The ISIC-2019 dataset was consolidated and expanded by merging labeled test data, yielding roughly 31,522 high-quality dermoscopic images. Stratified splits were created for training, validation, and testing to maintain class balance. To overcome class imbalance, minority classes were augmented until each contained approximately 3,500 images using a medically realistic Albumentations pipeline—applying flips, rotations, color jitter, coarse dropout, blurring, and elastic distortions. The ConvNeXt-Large model (from timm) was fine-tuned with AdamW optimizer, CosineAnnealingWarmRestarts scheduler, and Focal Loss (γ=4.0) with label smoothing (0.08). Custom class weights were applied to focus learning on high-risk lesion types (Melanoma ×4.2, SCC ×3.8, AK ×5.0). Grad-CAM visualization was integrated at the final ConvNeXt stage to verify the model’s focus on meaningful lesion regions.",

      "**Training & Validation Protocol:** The entire model was retrained from scratch for 130 epochs. Mixed-precision training, accumulated gradients, stochastic depth, and label smoothing ensured numerical stability and improved generalization. This pipeline consistently reproduced top validation and test metrics across multiple runs, confirming its reliability.",

      "**Key Results:** The final model achieved **88.30% validation accuracy** and **87.97% test accuracy**. Macro sensitivity was **86.27% (val)** and **86.31% (test)**, with an estimated **specificity of 98.08%**. Class-wise, Melanoma recall reached 87.7%, BCC 96.8%, and minority classes (VASC, DF) achieved strong precision/recall. These results show that a single ConvNeXt-Large model can achieve safer, more balanced performance than ensemble models reported in the literature.",

      "**Explainability & Clinical Safety:** Grad-CAM heatmaps confirmed that the model focuses on true lesion areas—edges, pigment variations, and internal texture—rather than background artifacts. Misclassifications tended to be conservative, overcalling benign lesions as malignant rather than missing real malignancies, which is the safer behavior in clinical diagnostics.",

      "**Impact Statement:** This project proves that a single, production-ready ConvNeXt-Large model can outperform complex ensembles in both clinical relevance and efficiency. It’s lightweight, interpretable, and practical for real-world healthcare use—ideal for telemedicine applications, triage systems, or dermatology decision support tools.",
    ],
    bullets: [
      "Achieved **88.30% validation accuracy** and **87.97% test accuracy** with a single ConvNeXt-Large model.",
      "Reached **macro sensitivity of 86.27%** and **specificity of 98.08%**, providing balanced, reliable clinical performance.",
      "**Outperformed ConvNeXt Ensembles** on sensitivity and specificity while drastically reducing computational cost.",
      "Integrated **Grad-CAM visualizations** to generate per-image interpretability, aligning focus with dermatological features such as asymmetry, border, and pigment variation.",
      "Designed a **clinically safe model** that prioritizes sensitivity to malignant cases, minimizing false negatives in high-risk categories.",
      "Developed a pipeline that’s efficient, explainable, and deployable—ready for telemedicine or hospital integration.",
    ],
  },

  // --- PAGE SECTIONS WITH IMAGES ---
  pagesInfoArr: [
    {
      title: "The Challenge — Dataset Imbalance & Visual Ambiguity",
      description:
        "The ISIC-2019 dataset, containing over 31,000 images across 8 lesion classes, was extremely imbalanced—dominated by benign nevi while malignant types were underrepresented. We expanded and cleaned the dataset, removed unknown labels, and visualized per-class distributions to inform augmentation and weighting strategies.",
      imgArr: [
        "/projects/skin-lesion/pie chart.png",
        "/projects/skin-lesion/sample_grid.png",
      ],
    },
    {
      title: "Training Pipeline — Targeted Augmentation & Loss Engineering",
      description:
        "To counteract imbalance, each class was augmented to roughly 3,500 samples using Albumentations. The augmentation strategy included rotations, color jitter, coarse dropout, Gaussian blur, and elastic transformations to simulate realistic dermatological variation. Focal Loss with manual class weighting (Melanoma ×4.2, SCC ×3.8, AK ×5.0) helped the model prioritize rare but clinically critical lesions.",
      imgArr: ["/projects/skin-lesion/augmentation_examples.png"],
    },
    {
      title: "Model & Performance — ConvNeXt-Large vs Ensembles",
      description:
        "The single ConvNeXt-Large model, pretrained on ImageNet-22K, achieved better sensitivity and specificity than heavy ensemble architectures while being simpler, faster, and more efficient to deploy. The model’s high recall on malignant classes demonstrates its reliability in real-world screening scenarios.",
      imgArr: ["/projects/skin-lesion/comparison.png"],
    },
    {
      title: "Explainability — Grad-CAM Visualizations",
      description:
        "Grad-CAM heatmaps visually confirmed that the model consistently attended to medically relevant regions. These heatmaps enhance clinical trust by showing the model’s 'reasoning'. Example visualizations display correct and borderline classifications with clear lesion-focused activation.",
      imgArr: [
        "/projects/skin-lesion/GRAD CAM.png",
        "/projects/skin-lesion/gradcam_grid.png",
      ],
    },
    {
      title: "Technical Appendix / Reproducibility",
      description:
        "Contains all implementation details—dataset configuration, augmentations, optimizer setup, and loss functions—ensuring reproducibility and transparency. This section could be displayed as an expandable technical panel in your portfolio.",
      imgArr: [],
    },
  ],

  // --- TECHNICAL APPENDIX ---
  technicalAppendix: {
    title: "Technical Details — Full Training Specification",
    content: [
      "**Dataset & Splits:** 31,522 dermoscopic images total. Stratified split: Train 70%, Validation 20%, Test 10%.",
      "**Augmentation Pipeline (Albumentations):** Horizontal/Vertical flip, RandomRotate90, ShiftScaleRotate, RandomGridShuffle (4×4), ColorJitter (brightness/contrast/saturation=0.2, p=0.7), GaussianNoise, GaussianBlur (limit 3–5 px), CoarseDropout (3–8 holes, 16–48 px), ElasticTransform, Optical Distortion.",
      "**Input Preprocessing:** 224×224 resolution, normalized using ImageNet mean and std.",
      "**Model Architecture:** ConvNeXt-Large (pretrained on ImageNet-22K) → Global Average Pooling → Linear(8) + Softmax.",
      "**Optimizer:** AdamW (lr = 5e-6, weight_decay = 1e-5) with gradient accumulation.",
      "**Scheduler:** CosineAnnealingWarmRestarts.",
      "**Loss Function:** Focal Loss (γ=4.0) with label smoothing (0.08) and boosted class weights (Melanoma×4.2, SCC×3.8, AK×5.0).",
      "**Training Setup:** 130 epochs, mixed precision, stochastic depth enabled, accumulated gradients for stability.",
      "**Evaluation Metrics:** Accuracy, Sensitivity (Recall), Specificity, Precision, F1 (macro & weighted).",
      "**Final Performance:** Validation — 88.30% acc, 86.27% sensitivity, 98.08% specificity; Test — 87.97% acc, 86.31% sensitivity.",
    ],
  },
},
{
  id: "parkinsons-disease-detection",
  companyName: "Parkinson’s Disease Detection from Voice Signals",
  type: "AI/ML",
  category: ["AI/ML", "Healthcare", "Signal Processing"],
  shortDescription:
    "Developed a robust ML pipeline to detect Parkinson’s disease from vocal biomarkers using XGBoost — achieving 92.3% accuracy and 95.5% F1-score with strong interpretability.",
  websiteLink: undefined,
  githubLink: undefined,
  techStack: ["Python", "Pandas", "Scikit-Learn", "XGBoost", "Matplotlib", "Seaborn"],
  startDate: null,
  endDate: null,
  companyLogoImg: "/projects/parkinsons/parkinsons-logo.png",

  // 🔹 Hero section for top-of-page metrics
  heroInfo: {
    headline:
      "Accurate, explainable detection of Parkinson’s disease using acoustic features — powered by XGBoost.",
    keyMetrics: [
      { label: "Accuracy", value: "92.3%" },
      { label: "Recall", value: "97.5%" },
      { label: "F1-Score", value: "95.5%" },
    ],
  },

  // 🔹 Main detailed case study sections
  descriptionSections: [
    {
      title: "Project Summary",
      content: [
        "Parkinson’s disease affects millions worldwide, yet early diagnosis remains difficult without costly clinical assessments. This project explores how subtle vocal biomarkers — captured through sustained phonation — can be used for accurate and interpretable disease prediction.",
        "Using the classic Parkinson’s voice dataset, I built a complete ML pipeline that preprocesses data, benchmarks multiple algorithms, and identifies an optimal model. The final **XGBoost classifier** achieved 92.3% accuracy and 95.5% F1-score, outperforming other tested models like SVM, KNN, and Random Forest.",
      ],
    },
    {
      title: "Problem & Motivation",
      content: [
        "Traditional diagnosis of Parkinson’s relies on clinical evaluation and motor assessments, which are time-intensive and require specialists.",
        "**Goal:** Develop a low-cost, non-invasive system that can detect Parkinson’s disease from simple voice recordings, helping with early screening and remote diagnosis.",
      ],
    },
    {
      title: "Hypothesis & Approach",
      content: [
        "I hypothesized that **acoustic features** (e.g., jitter, shimmer, frequency variations) are strong indicators of vocal irregularities associated with Parkinson’s disease.",
        "The pipeline performs:",
        "• **Feature standardization** using `StandardScaler` to stabilize training.",
        "• **Exploratory analysis** and correlation heatmaps to detect feature relationships.",
        "• **Model benchmarking** across multiple algorithms — KNN, Logistic Regression, SVM, Random Forest, and XGBoost.",
        "• **Cross-validation and evaluation** using accuracy, precision, recall, and F1 metrics.",
      ],
    },
    {
      title: "1. Data Preparation & EDA",
      content: [
        "Loaded the canonical `parkinsons.data` dataset containing 195 samples and 22 features derived from sustained phonation recordings.",
        "Performed exploratory data analysis to inspect distributions, outliers, and feature correlations using Seaborn heatmaps.",
        "Verified dataset integrity (no missing values) and visualized class balance to inform modeling decisions.",
      ],
    },
    {
      title: "2. Model Training & Evaluation",
      content: [
        "Trained five classical ML models — **KNN**, **Logistic Regression**, **SVM**, **Random Forest**, and **XGBoost** — with stratified train/test splitting to preserve label balance.",
        "Used cross-validation for fair evaluation and consistent metrics across runs.",
        "Below are test accuracies reported from the notebook:",
        "• KNN — 86.44%  |  Logistic Regression — 84.74%  |  SVM — 83.05%  |  Random Forest — 89.83%  |  **XGBoost — 92.30%**",
        "The **XGBoost model** achieved the highest accuracy and recall, making it the final selected model.",
      ],
    },
    {
      title: "3. Explainability & Insights",
      content: [
        "Feature importance analysis confirmed that the model focuses on physiologically meaningful vocal attributes like **MDVP:Jitter(%)**, **MDVP:Shimmer(dB)**, and **spread1/spread2**.",
        "Correlation analysis validated relationships among these features and helped ensure model stability.",
        "Gradual feature elimination experiments showed robustness — accuracy stayed within ±1% even when less informative features were dropped.",
      ],
    },
    {
      title: "4. Impact & Applications",
      content: [
        "The final XGBoost-based model can be deployed as a **screening tool** for clinicians and telemedicine applications — requiring only basic voice samples as input.",
        "It demonstrates that **lightweight classical ML methods** can rival deep learning in structured biomedical domains, with higher interpretability and lower computational cost.",
        "With 97.5% recall, the system minimizes false negatives — a key priority in medical screening where missed diagnoses are critical.",
      ],
    },
  ],

  // 🔹 Key achievements (summary bullets)
  bullets: [
    "Built a full ML pipeline for Parkinson’s detection from voice-derived features.",
    "Benchmarked five models; achieved 92.3% accuracy and 95.5% F1-score using XGBoost.",
    "Performed EDA, correlation, and feature importance analysis for explainability.",
    "Demonstrated medically meaningful feature reliance, boosting clinical trust.",
    "Delivered a lightweight, deployable model for early screening and telehealth use.",
  ],

  // 🔹 Case study sections with visuals
  pagesInfoArr: [
    {
      title: "EDA & Correlation Analysis",
      description:
        "Visualized data distributions, pairwise correlations, and target imbalance to identify influential features for modeling.",
      imgArr: [
        "/projects/parkinsons/eda_overview.png",
        "/projects/parkinsons/correlation_heatmap.png",
      ],
    },
    {
      title: "Model Benchmarking",
      description:
        "Compared the performance of five classical ML models and selected XGBoost based on accuracy, recall, and F1 metrics.",
      imgArr: [
        "/projects/parkinsons/model_comparison.png",
        "/projects/parkinsons/confusion_matrix.png",
      ],
    },
    {
      title: "Feature Importance & Explainability",
      description:
        "Displayed top contributing acoustic features (e.g., Jitter, Shimmer, Spread1) that drive model predictions, validating interpretability.",
      imgArr: ["/projects/parkinsons/feature_importance.png"],
    },
    {
      title: "Deployment Readiness",
      description:
        "Saved the trained XGBoost model and preprocessing pipeline artifacts, enabling reproducible inference in Flask or FastAPI environments.",
      imgArr: ["/projects/parkinsons/model_export.png"],
    },
  ],

  // 🔹 Technical appendix (for reproducibility)
  technicalAppendix: {
    title: "Technical Appendix — Model & Training Details",
    content: [
      "Dataset: 195 samples × 22 features (acoustic signals from sustained phonation).",
      "Preprocessing: StandardScaler normalization on continuous features.",
      "Train-test split: 80/20 with stratification.",
      "Models Benchmarked: KNN, Logistic Regression, SVM (RBF), Random Forest, XGBoost.",
      "Best Model: XGBoost — n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=42.",
      "Test Results: Accuracy = 92.3%, Precision = 94.5%, Recall = 97.5%, F1 = 95.5%.",
      "Feature Importance: Jitter(%), Shimmer(dB), spread1, spread2, and PPE among top indicators.",
      "Artifacts Saved: model.pkl and columns.pkl for reproducible inference.",
      "Future Work: Expand dataset, apply cross-patient validation, integrate with voice recording front-end for real-world deployment.",
    ],
  },
},
// ...
{
  id: "cholo-pothik",
  companyName: "Cholo Pothik (Travel Management System)",
  type: "Web/Software",
  category: ["Web Dev", "Full Stack", "Travel"],
  shortDescription: "A full-stack travel platform in PHP/MySQL. I built the complete authentication system and the entire admin dashboard for content management.",
  websiteLink: undefined, // <-- Add link later if you have one
  githubLink: "https://github.com/Azmain-Khan-Alve/Cholo-Pothik", // From your report
  techStack: ["PHP", "MySQL", "HTML 5", "CSS 3"],
  startDate: null, // As requested
  endDate: null,   // As requested
  companyLogoImg: "/projects/cholo-pothik-logo.png", // <-- IMPORTANT: Add a logo to /public/projects/ and change this
  
  // --- PASTE THIS NEW CASE STUDY ---
  descriptionDetails: {
    paragraphs: [
      "A comprehensive, full-stack travel portal built from scratch in PHP and MySQL. The platform allows users to browse and book travel packages, while providing a secure backend for administrators to manage all site content.",
      "My core contribution was building the project's 'brain.' I developed the entire secure authentication system (user signup, login, admin-only access) and the complete admin dashboard, which gives administrators full CRUD (Create, Read, Update, Delete) control over hotels, places, and travel agents."
    ],
    bullets: [
      "Developed the complete user and admin authentication/session logic in PHP.",
      "Built the full admin-side dashboard for viewing all site data (customers, bookings, etc.).",
      "Engineered the backend logic (Insert, Delete, Update) for managing all travel content.",
      "Personally responsible for the 'Authentication' and 'Admin Dashboard' modules as outlined in the project report."
    ],
  },
  pagesInfoArr: [
    {
      title: "Secure Authentication (Login/Signup)",
      description: "Backend code snippets for user signup and secure login (Fig. 17 & 18 from report).",
      imgArr: [
        "/projects/cholo-pothik/fig17_signin.png",
        "/projects/cholo-pothik/fig18_signup.png"
      ]
    },
    {
      title: "Admin Dashboard Operations (Backend)",
      description: "Backend logic for the admin's content management operations (Fig. 19 from report).",
      imgArr: [
        "/projects/cholo-pothik/fig19_admin_op.png"
      ]
    },
    {
      title: "Core Booking & Search Logic (Backend)",
      description: "Backend snippets for the site's core search and booking functionality (Fig. 20 & 22 from report).",
      imgArr: [
        "/projects/cholo-pothik/fig20_search.png",
        "/projects/cholo-pothik/fig22_booking.png"
      ]
    }
  ]
  // --- END OF NEW CASE STUDY ---
},
// ...
];

export const featuredProjects = Projects.slice(0, 3);









// =================================================================================



// import { ValidCategory, ValidExpType, ValidSkills } from "./constants";

// interface PagesInfoInterface {
//   title: string;
//   imgArr: string[];
//   description?: string;
// }

// interface DescriptionDetailsInterface {
//   paragraphs: string[];
//   bullets: string[];
// }

// export interface ProjectInterface {
//   id: string;
//   type: ValidExpType;
//   companyName: string;
//   category: ValidCategory[];
//   shortDescription: string;
//   websiteLink?: string;
//   githubLink?: string;
//   techStack: ValidSkills[];
//   startDate: Date | null;
//   endDate: Date | null;
//   companyLogoImg: any;
//   descriptionDetails: DescriptionDetailsInterface;
//   pagesInfoArr: PagesInfoInterface[];
// }

// export const Projects: ProjectInterface[] = [

//   {
//     id: "diabetes-risk-prediction",
//     companyName: "Clinical Data-Driven Diabetes Risk Prediction",
//     type: "AI/ML",
//     category: ["AI/ML", "Classical ML", "Healthcare"],
//     shortDescription: "End-to-end ML pipeline for diabetes risk. Achieved 97.1% accuracy and 0.964 ROC-AUC using XGBoost, with deep EDA and feature analysis.",
//     websiteLink: "https://huggingface.co/spaces/Alve69/diabetes-risk-prediction",
//     githubLink: "https://github.com/Azmain-Khan-Alve/Diabetes-Risk-Prediction",
//     techStack: ["Scikit-Learn", "XGBoost", "Pandas", "Matplotlib", "Seaborn"],
//     startDate: null, // Assumed date
//     endDate: null, // Assumed date
//     companyLogoImg: "/projects/diabetes-logo.png", // We will add this image later
    
//     // Leave empty for now
//     pagesInfoArr: [], 
//     descriptionDetails: {
//       paragraphs: [],
//       bullets: [],
//     },
//   },
//   {
//   id: "parkinsons-disease-detection",
//   companyName: "Early Detection of Parkinson's Disease",
//   type: "AI/ML",
//   category: ["AI/ML", "Classical ML", "Healthcare"],
//   shortDescription: "Evaluated ML models on voice data (UCI dataset). XGBoost (92% Acc) and Random Forest (89% Acc) were top models. Identified key vocal features.",
//   websiteLink: undefined,
//   githubLink: "https://github.com/Azmain-Khan-Alve/Parkinson-s-Disease-Detection",
//   techStack: ["Scikit-Learn", "XGBoost", "Random Forest", "Pandas"],
//   startDate: null, // Assumed date
//   endDate: null, // Assumed date
//   companyLogoImg: "/projects/parkinsons-logo.png", // We will add this image later
  
//   // Leave empty for now
//   pagesInfoArr: [], 
//   descriptionDetails: {
//     paragraphs: [],
//     bullets: [],
//   },
// },

// {
//   id: "cholo-pothik",
//   companyName: "Cholo Pothik",
//   type: "Web/Software",
//   category: ["Web Dev", "Full Stack", "Travel"],
//   shortDescription: "Full-stack Travel Management System with separate user and admin interfaces.",
//   websiteLink: undefined, // <-- Add link later if you have one
//   githubLink: "https://github.com/Azmain-Khan-Alve/Cholo-Pothik", 
//   techStack: ["PHP", "MySQL", "HTML 5", "CSS 3"],
//   startDate: null, // <-- IMPORTANT: Change this date
//   endDate: null, // <-- IMPORTANT: Change this date
//   companyLogoImg: "/projects/placeholder.png", // <-- IMPORTANT: Add a logo to /public/projects/ and change this path
  
//   pagesInfoArr: [], 
//   descriptionDetails: {
//     paragraphs: [
//       "A full-stack Travel Management System built from scratch, featuring separate, secure interfaces for both users and administrators."
//     ],
//     bullets: [
//       "User Interface: Features user authentication, destination search, hotel booking capabilities, and a feedback system.",
//       "Admin Interface: Provides full Create, Read, Update, and Delete (CRUD) operations for managing places, hotels, and travel agents."
//     ],
//   },
// },
// {
//   id: "cloud-pharma",
//   companyName: "Cloud Pharma",
//   type: "Web/Software",
//   category: ["Web Dev", "Full Stack", "E-commerce"],
//   shortDescription: "Online medicine sales platform with an admin dashboard and business insights.",
//   websiteLink: undefined, // <-- Add link later if you have one
//   githubLink: "https://github.com/Azmain-Khan-Alve/Cloud-Pharma", // <-- Add your GitHub link here
//   techStack: ["Django", "PostgreSQL", "Bootstrap"],
//   startDate: new Date("2023-06-01"), // <-- IMPORTANT: Change this date
//   endDate: new Date("2023-10-01"), // <-- IMPORTANT: Change this date
//   companyLogoImg: "/projects/placeholder.png", // <-- IMPORTANT: Add a logo to /public/projects/ and change this path
  
//   pagesInfoArr: [], 
//   descriptionDetails: {
//     paragraphs: [
//       "An online medicine sales platform developed using the Django framework, designed for secure and efficient e-commerce operations."
//     ],
//     bullets: [
//       "Features secure authentication, a complete medicine catalog, a full cart and checkout system, and delivery tracking.",
//       "Includes a comprehensive admin dashboard to manage medicines, customers, and orders, with integrated statistics for business insights."
//     ],
//   },
// },
// // --- END OF NEW PROJECTS ---

// // {
// //   id: "built-design",
// //   companyName: "Builtdesign",
// //   type: "Professional",
// //   category: ["Web Dev", "Full Stack", "UI/UX"],
// //   shortDescription:
// //     "Developed and optimized a high-performing website catering to over 4000 users, emphasizing efficiency and maintainability.",
// //   websiteLink: "https://builtdesign.in",
// //   techStack: [
// //     "Next.js",
// //     "React",
// //     "Node.js",
// //     "MongoDB",
// //     "GraphQL",
// //     "Nest.js",
// //     "Typescript",
// //   ],
// //   startDate: new Date("2021-07-01"),
// //   endDate: new Date("2022-07-01"),
// //   companyLogoImg: "/projects/builtdesign/logo.png",
// //   pagesInfoArr: [
// //     {
// //       title: "Landing Page",
// //       description:
// //         "Modern and responsive landing page showcasing company services and portfolio",
// //       imgArr: [
// //         "/projects/builtdesign/landing_1.webp",
// //         "/projects/builtdesign/landing_3.webp",
// //         "/projects/builtdesign/landing_5.webp",
// //         "/projects/builtdesign/landing_6.webp",
// //         "/projects/builtdesign/landing_2.webp",
// //         "/projects/builtdesign/landing_4.webp",
// //       ],
// //     },
// //     {
// //       title: "Custom PDF Reader and optimizer",
// //       description:
// //         "Specialized PDF viewer with optimization features for improved performance and user experience",
// //       imgArr: ["/projects/builtdesign/pdf_opt.webp"],
// //     },
// //     {
// //       title: "Clients Dashboard",
// //       description:
// //         "Comprehensive client portal with project tracking, document management, and communication tools",
// //       imgArr: [
// //         "/projects/builtdesign/cli_dashboard_1.webp",
// //         "/projects/builtdesign/cli_dashboard_2.webp",
// //         "/projects/builtdesign/cli_dashboard_3.webp",
// //       ],
// //     },
// //     {
// //       title: "Admin Dashboard",
// //       description:
// //         "Powerful administrative interface for managing users, projects, and system settings",
// //       imgArr: ["/projects/builtdesign/logo.png"],
// //     },
// //   ],
// //   descriptionDetails: {
// //     paragraphs: [
// //       "During my time at Builtdesign, I had the opportunity to work on a dynamic and user-focused project that involved designing and optimizing a website catering to a user base of over 4000 individuals. My role as a full-stack web developer was to ensure a seamless experience for users by creating an efficient and maintainable platform.",
// //       "I collaborated closely with the product team to integrate cutting-edge features, employing technologies like Next.js and React with TypeScript for captivating front-end experiences. Additionally, I contributed significantly to the backend by utilizing Node.js, MongoDB, and GraphQL to design robust APIs and ensure smooth system functionality.",
// //       "This experience allowed me to enhance my skills in various areas of web development and deliver a high-quality product. I gained proficiency in front-end technologies such as Material UI and Tailwind CSS, as well as backend technologies including Nest.js and MySQL. The project's success in catering to a large user base and providing an intuitive user interface has further motivated me to pursue excellence in web development.",
// //     ],
// //     bullets: [
// //       "Developed and optimized a high-performing website catering to over 4000 users.",
// //       "Collaborated closely with the product team to implement cutting-edge features.",
// //       "Created an intuitive admin dashboard to efficiently manage and announce contest winners.",
// //       "Leveraged Next.js, React with TypeScript for captivating front-end experiences.",
// //       "Utilized Node.js, MongoDB, and GraphQL to design and manage databases.",
// //     ],
// //   },
// // },
// //   {
// //     id: "the-super-quotes",
// //     companyName: "The Super Quotes",
// //     type: "Professional",
// //     category: ["Mobile Dev", "Full Stack", "UI/UX"],
// //     shortDescription:
// //       "Elevated The Super Quotes app with JavaScript, React Native, APIs, Redux magic, and Google Play Store debut.",
// //     websiteLink:
// //       "https://play.google.com/store/apps/details?id=com.thesuperlife",
// //     techStack: ["React Native", "Node.js", "MongoDB", "Javascript"],
// //     startDate: new Date("2021-07-01"),
// //     endDate: new Date("2022-07-01"),
// //     companyLogoImg: "/projects/superquotes/logo.png",
// //     pagesInfoArr: [
// //       {
// //         title: "Quotes View Page",
// //         description:
// //           "Elegantly designed quotes display with customizable themes and sharing options",
// //         imgArr: ["/projects/superquotes/app_2.webp"],
// //       },
// //       {
// //         title: "Quotes Download Component",
// //         description:
// //           "Feature allowing users to download quotes as beautiful images for social media sharing",
// //         imgArr: [
// //           "/projects/superquotes/app_4.webp",
// //           "/projects/superquotes/app_7.webp",
// //         ],
// //       },
// //       {
// //         title: "Account Management",
// //         description:
// //           "User profile management with favorites, history, and personalization settings",
// //         imgArr: ["/projects/superquotes/app_6.webp"],
// //       },
// //       {
// //         title: "Interest Selection and Update Page",
// //         description:
// //           "Interactive interface for users to select and update their quote preferences and interests",
// //         imgArr: [
// //           "/projects/superquotes/app_1.webp",
// //           "/projects/superquotes/app_3.webp",
// //         ],
// //       },
// //       {
// //         title: "Responsiveness",
// //         description:
// //           "Adaptive design ensuring optimal user experience across various device sizes and orientations",
// //         imgArr: ["/projects/superquotes/app_5.webp"],
// //       },
// //     ],
// //     descriptionDetails: {
// //       paragraphs: [
// //         `Venturing into the world of creativity at The Super Quotes was an exhilarating journey. As a passionate developer, I led the charge in crafting a compelling application from inception to completion, using the dynamic duo of JavaScript and React Native.`,
// //         `The heart of my achievement lay in the seamless integration of APIs, threading a tapestry of data flow that propelled the application's functionality to new heights.`,
// //         `With the wizardry of Redux, I choreographed a symphony of state management and performance optimization, orchestrating a ballet of responsiveness that wowed users with every interaction.`,
// //         `A crescendo awaited as I unveiled the culmination of my work on the grand stage of the Google Play Store. The app's debut marked an epoch, opening doors to an expansive audience eager to embrace the charm of The Super Quotes.`,
// //       ],
// //       bullets: [
// //         "Led the end-to-end development of a captivating application using JavaScript and React Native.",
// //         "Championed the integration of APIs, harmonizing data flow and enhancing application functionality.",
// //         "Conducted Redux magic to ensure state management and optimize performance, delivering a mesmerizing user experience.",
// //         "Premiered the application on the Google Play Store, capturing hearts and expanding its user base.",
// //       ],
// //     },
// //   },
// //   {
// //     id: "apex-shopping",
// //     companyName: "Apex Shopping App",
// //     type: "Personal",
// //     category: ["Mobile Dev", "Full Stack", "UI/UX"],
// //     shortDescription:
// //       "Developed a feature-rich mobile shopping application with admin panel, user authentication, and seamless product management using React Native and Firebase.",
// //     githubLink: "https://github.com/namanbarkiya/apex-shopping-app",
// //     techStack: ["React Native", "Javascript", "Redux", "Node.js", "express.js"],
// //     startDate: new Date("2021-07-14"),
// //     endDate: new Date("2022-07-01"),
// //     companyLogoImg: "/projects/apex/logo.png",
// //     pagesInfoArr: [
// //       {
// //         title: "Splash Screen",
// //         description: "Custom animated splash screen with app branding",
// //         imgArr: ["/projects/apex/app_7.webp"],
// //       },
// //       {
// //         title: "Login/Signup Authentication",
// //         description: "Secure user authentication system with Firebase",
// //         imgArr: ["/projects/apex/app_1.webp"],
// //       },
// //       {
// //         title: "All Products Explore Screen",
// //         description: "Interactive product browsing with categories and filters",
// //         imgArr: ["/projects/apex/app_3.webp"],
// //       },
// //       {
// //         title: "Admin Panel",
// //         description:
// //           "Comprehensive admin dashboard for product and order management",
// //         imgArr: ["/projects/apex/app_4.webp", "/projects/apex/app_6.webp"],
// //       },
// //       {
// //         title: "Sidenav Navigation",
// //         description: "Intuitive side navigation for easy app navigation",
// //         imgArr: ["/projects/apex/app_5.webp"],
// //       },
// //       {
// //         title: "Firebase Database",
// //         description:
// //           "Real-time database structure for efficient data management",
// //         imgArr: ["/projects/apex/db.webp"],
// //       },
// //     ],
// //     descriptionDetails: {
// //       paragraphs: [
// //         "The Apex Shopping App represents a comprehensive mobile e-commerce solution that I developed from the ground up using React Native and Firebase. This project showcases my ability to create a full-featured shopping application with both user and admin functionalities.",
// //         "The application features a robust authentication system, allowing users to securely sign up and log in. The product exploration interface is designed with user experience in mind, incorporating smooth navigation and intuitive filtering options.",
// //         "One of the key highlights is the admin panel, which provides complete control over product management, order processing, and inventory tracking. The integration with Firebase ensures real-time data synchronization and reliable data persistence.",
// //         "The app's architecture emphasizes scalability and performance, utilizing Redux for state management and following best practices for mobile app development. The UI/UX design focuses on providing a seamless shopping experience across different device sizes.",
// //       ],
// //       bullets: [
// //         "Implemented secure user authentication and authorization using Firebase",
// //         "Designed and developed an intuitive product browsing and shopping cart system",
// //         "Created a comprehensive admin panel for product and order management",
// //         "Integrated real-time data synchronization using Firebase Database",
// //         "Implemented state management using Redux for optimal performance",
// //         "Designed responsive UI components following mobile-first principles",
// //         "Incorporated smooth animations and transitions for enhanced user experience",
// //       ],
// //     },
// //   },
// //   {
// //     id: "builtdesign-blogs",
// //     companyName: "Builtdesign Blogs",
// //     type: "Professional",
// //     category: ["Web Dev", "Full Stack", "UI/UX"],
// //     shortDescription:
// //       "Crafted Builtdesign's vibrant Blogs Website using Netlify CMS and React for engaging content experiences.",
// //     websiteLink: "https://blog.builtdesign.in",
// //     techStack: ["Next.js", "React", "Node.js", "MongoDB", "Typescript"],
// //     startDate: new Date("2022-03-01"),
// //     endDate: new Date("2022-07-01"),
// //     companyLogoImg: "/projects/builtdesign-blogs/logo.png",
// //     pagesInfoArr: [
// //       {
// //         title: "Blog Landing Page",
// //         description:
// //           "Modern and responsive landing page showcasing featured articles",
// //         imgArr: ["/projects/builtdesign-blogs/blog_2.webp"],
// //       },
// //       {
// //         title: "Blog Listing",
// //         description:
// //           "Organized display of all blog posts with search and filtering",
// //         imgArr: ["/projects/builtdesign-blogs/blog_3.webp"],
// //       },
// //       {
// //         title: "Category Navigation",
// //         description: "Intuitive category-based navigation system",
// //         imgArr: ["/projects/builtdesign-blogs/blog_1.webp"],
// //       },
// //       {
// //         title: "Article View",
// //         description:
// //           "Clean and readable article layout with rich media support",
// //         imgArr: [
// //           "/projects/builtdesign-blogs/blog_4.webp",
// //           "/projects/builtdesign-blogs/blog_5.webp",
// //         ],
// //       },
// //     ],
// //     descriptionDetails: {
// //       paragraphs: [
// //         "As part of the Builtdesign platform, I developed a sophisticated blog website that serves as a content hub for the company's thought leadership and industry insights. The project leveraged Next.js and React to create a fast, SEO-friendly platform.",
// //         "The blog platform features a modern, responsive design that prioritizes readability and user engagement. I implemented a robust content management system using Netlify CMS, enabling the content team to easily publish and manage blog posts.",
// //         "The architecture includes server-side rendering for optimal performance and SEO, while MongoDB provides flexible content storage. TypeScript ensures code reliability and maintainability throughout the application.",
// //         "Key features include category-based navigation, search functionality, and a rich text editor for content creation. The platform supports various content types including images, code snippets, and embedded media.",
// //       ],
// //       bullets: [
// //         "Developed a modern blog platform using Next.js and React with TypeScript",
// //         "Implemented Netlify CMS for efficient content management",
// //         "Created a responsive design that prioritizes readability and user engagement",
// //         "Built server-side rendering for optimal performance and SEO",
// //         "Integrated MongoDB for flexible content storage and management",
// //         "Developed category-based navigation and search functionality",
// //         "Implemented rich text editing capabilities for content creation",
// //       ],
// //     },
// //   },
// //   {
// //     id: "portfolio-card",
// //     companyName: "Portfolio Card",
// //     type: "Personal",
// //     category: ["Web Dev", "Frontend", "3D Modeling"],
// //     shortDescription:
// //       "Forged an immersive 3D Portfolio Card utilizing the prowess of Three.js and Blender, where art and technology converge in an interactive masterpiece.",
// //     websiteLink: "https://card.namanbarkiya.xyz/",
// //     githubLink: "https://github.com/namanbarkiya/3d-portfolio-card",
// //     techStack: ["React", "Javascript", "HTML 5", "CSS 3"],
// //     startDate: new Date("2022-03-01"),
// //     endDate: new Date("2022-07-01"),
// //     companyLogoImg: "/projects/card/logo.png",
// //     pagesInfoArr: [
// //       {
// //         title: "Card Views",
// //         description: "Front and back views of the interactive 3D card",
// //         imgArr: ["/projects/card/card_2.webp", "/projects/card/card_3.webp"],
// //       },
// //       {
// //         title: "Interactive Elements",
// //         description:
// //           "Custom links embedded in the 3D model with interactive animations",
// //         imgArr: ["/projects/card/card_1.webp"],
// //       },
// //       {
// //         title: "3D Model Development",
// //         description: "Blender project showcasing the model creation process",
// //         imgArr: ["/projects/card/card_4.webp"],
// //       },
// //     ],
// //     descriptionDetails: {
// //       paragraphs: [
// //         "In my personal, I've ventured into the world of creativity, fashioning a distinctive portfolio card through the utilization of Three.js.",
// //         "This portfolio card transcends convention; it emerges as a captivating 3D model, adorned with meticulous lighting arrangements that conjure a spellbinding visual journey.",
// //         "To materialize this concept, I've harnessed the combined potential of Three.js and Blender, orchestrating a meticulous crafting of the central 3D model that serves as the cornerstone of the card's allure.",
// //         "Yet, the allure extends beyond aesthetics. I've ingeniously interwoven custom links directly into the fabric of Three.js components. Through the creation and seamless integration of novel components, these additions elegantly rest upon the card's surface, mirroring its rotations and delivering an interactive dimension to my portfolio.",
// //         "The portfolio card itself is an opus of motion, perpetually swaying in an auto-rotational dance that unfurls its multifaceted essence. As an enhancement, I've introduced an instinctive user interaction element. A simple, intuitive drag of the card in specific directions grants viewers a comprehensive vantage, enabling exploration from every conceivable angle.",
// //         "At its core, my personal epitomizes technical finesse, artistic expression, and interactive design. The amalgamation of Three.js, Blender's prowess, and the innovation of component integration has birthed not only a portfolio card, but a dynamic encounter leaving an indelible imprint on all who partake.",
// //       ],
// //       bullets: [
// //         "Conceptualized and realized a distinct portfolio card using Three.js, highlighting creative exploration.",
// //         "Crafted a mesmerizing 3D model enhanced by thoughtful lighting arrangements, resulting in a captivating visual voyage.",
// //         "Leveraged the synergy of Three.js and Blender to meticulously sculpt and refine the central 3D model, embodying meticulous attention to detail.",
// //         "Innovatively integrated custom links within Three.js components, introducing an interactive layer via seamlessly incorporated new elements.",
// //         "Enabled an auto-rotating feature for the portfolio card, perpetually showcasing its various facets to observers.",
// //         "Introduced an instinctual user interaction mechanism, allowing viewers to comprehensively explore the card's dimensions through simple, intuitive dragging motions.",
// //         "Represented a fusion of technical prowess, artistic ingenuity, and interactive design in a project that reshapes the boundaries of conventional portfolio representation.",
// //       ],
// //     },
// //   },
// //   {
// //     id: "cirql-dashboard",
// //     companyName: "Cirql Dashboard",
// //     type: "Personal",
// //     category: ["Web Dev", "Frontend", "UI/UX"],
// //     shortDescription:
// //       "Created a dashboard project using React and Tailwind CSS, focusing on UI design and routing implementation.",
// //     websiteLink: "https://cirql-ui.namanbarkiya.xyz/",
// //     techStack: ["React", "Tailwind CSS", "Google Auth"],
// //     startDate: new Date("2023-01-01"),
// //     endDate: new Date("2023-02-15"),
// //     companyLogoImg: "/projects/cirql/logo.png",
// //     pagesInfoArr: [
// //       {
// //         title: "Dashboard Home",
// //         description:
// //           "Main dashboard view with analytics widgets and data visualization",
// //         imgArr: ["/projects/cirql/web_1.png", "/projects/cirql/web_2.png"],
// //       },
// //       {
// //         title: "Profile Page",
// //         description:
// //           "User profile management interface with customization options",
// //         imgArr: ["/projects/cirql/web_3.png", "/projects/cirql/web_4.png"],
// //       },
// //     ],
// //     descriptionDetails: {
// //       paragraphs: [
// //         "For the 'Cirql Dashboard' personal, I aimed to enhance my UI design skills and deepen my understanding of routing within a React application.",
// //         "I utilized React and Tailwind CSS to craft an intuitive dashboard interface that provides users with an organized overview of data and functionalities. The UI components were thoughtfully designed to ensure a seamless user experience.",
// //         "Incorporating Google Sign-In Authentication further fortified the project by adding a layer of security and convenience. Users are required to authenticate before accessing certain routes, ensuring the safety of sensitive information.",
// //         "The routing system was meticulously implemented to enable smooth navigation between different sections of the dashboard, simulating real-world use cases.",
// //         "Through this project, I've gained valuable insights into UI/UX design principles and the implementation of secure and efficient routing in React applications.",
// //       ],
// //       bullets: [
// //         "Created a user-friendly dashboard project using React and Tailwind CSS.",
// //         "Implemented Google Sign-In Authentication to ensure secure access to sensitive routes.",
// //         "Designed UI components to provide an intuitive and visually pleasing experience.",
// //         "Focused on implementing a smooth routing system to simulate real-world use cases.",
// //         "Enhanced my skills in UI design, routing, and component architecture.",
// //       ],
// //     },
// //   },
// //   {
// //     id: "inscript-hindi-typing",
// //     companyName: "Inscript Hindi Typing",
// //     type: "Personal",
// //     category: ["Web Dev", "UI/UX"],
// //     shortDescription:
// //       "Developed a user-friendly website for Inscript Hindi typing, addressing the need for a simple tool for Hindi writers to convey data digitally.",
// //     websiteLink: "https://hindityping.namanbarkiya.xyz",
// //     githubLink: "https://github.com/namanbarkiya/inscript-hindi-keyboard",
// //     techStack: ["HTML 5", "CSS 3", "Javascript"],
// //     startDate: new Date("2022-05-01"),
// //     endDate: new Date("2022-06-15"),
// //     companyLogoImg: "/projects/hindi-keyboard/logo.png",
// //     pagesInfoArr: [
// //       {
// //         title: "Typing Interface",
// //         description: "Minimal and user-friendly Inscript Hindi typing area",
// //         imgArr: ["/projects/hindi-keyboard/web_1.png"],
// //       },
// //       {
// //         title: "Copy and Download the file",
// //         description:
// //           "Export functionality allowing users to copy text or download as a document file",
// //         imgArr: [
// //           "/projects/hindi-keyboard/web_2.png",
// //           "/projects/hindi-keyboard/web_3.png",
// //         ],
// //       },
// //     ],
// //     descriptionDetails: {
// //       paragraphs: [
// //         "The 'Inscript Hindi Typing Website' project emerged from the need to provide a simple and accessible tool for Hindi writers, especially those in digital news and media, who wished to convey data in Hindi.",
// //         "Recognizing the challenges posed by complex software in the market, I set out to create a minimalistic typing area that catered to the needs of a vast community of Hindi typists in India.",
// //         "The project was designed to address the specific requirements of users familiar with the Inscript keyboard layout, mapping English and Hindi alphabets for seamless typing. The intuitive interface allowed users to effortlessly switch between languages, streamlining the process of content creation.",
// //         "Leveraging HTML and CSS, I crafted the website's UI to ensure a user-friendly experience. Additionally, Local Storage was utilized to enable users to save and retrieve their work, enhancing convenience and productivity.",
// //         "The website's focus on user experience and simplicity proved to be a key factor in its popularity among Hindi writers. By offering a tool that reduced the barriers to entry, I contributed to the digital empowerment of Hindi typists who previously faced challenges in conveying their message effectively.",
// //         "This project marked one of my initial forays into web development and highlighted the transformative potential of technology in addressing real-world challenges.",
// //       ],
// //       bullets: [
// //         "Developed a user-friendly website for Inscript Hindi typing.",
// //         "Catered to the needs of Hindi writers in digital news and media.",
// //         "Created a minimalistic and intuitive typing interface for the Inscript keyboard layout.",
// //         "Mapped English and Hindi alphabets to provide a seamless typing experience.",
// //         "Utilized HTML and CSS to design a user-friendly UI.",
// //         "Implemented Local Storage to enable users to save and retrieve their work.",
// //         "Contributed to the digital empowerment of Hindi typists by offering a simple tool.",
// //         "Marked one of my first web development projects, showcasing technology's potential for addressing real-world needs.",
// //       ],
// //     },
// //   },
// ];

// export const featuredProjects = Projects.slice(0, 3);
