import { Icons } from "@/components/common/icons";

export interface skillsInterface {
  name: string;
  description: string;
  rating: number; // We keep this for sorting
  imageUrl: string; // This is YOUR new idea. It's a string, not a function.
  // We've added your two new categories to this list
  category: "Languages" | "Frameworks" | "Python Libraries" | "Database" | "Cloud & MLOps" | "Version Control" | "Office Skills";
}

export const skillsUnsorted: skillsInterface[] = [
  // Languages
  {
    name: "Python",
    description: "The primary language for data science and AI development.",
    rating: 5,
    imageUrl: "/logos/python.svg",
    category: "Languages",
  },

  // Frameworks
  {
    name: "PyTorch",
    description: "A leading deep learning framework for research and production.",
    rating: 5,
    imageUrl: "/logos/pytorch.svg",
    category: "Frameworks",
  },
  {
    name: "TensorFlow",
    description: "A comprehensive ecosystem for building and deploying ML models.",
    rating: 5,
    imageUrl: "/logos/tensorflow.svg",
    category: "Frameworks",
  },
  {
    name: "Keras",
    description: "A high-level neural networks API, running on top of TensorFlow.",
    rating: 4,
    imageUrl: "/logos/keras.svg",
    category: "Frameworks",
  },
  {
    name: "Flask",
    description: "A lightweight web framework for building model APIs.",
    rating: 4,
    imageUrl: "/logos/flask.svg",
    category: "Frameworks",
  },
  { // NEWLY ADDED
    name: "Django",
    description: "A high-level Python web framework for rapid development.",
    rating: 4,
    imageUrl: "/logos/django.svg",
    category: "Frameworks",
  },
  { // NEWLY ADDED
    name: "Laravel",
    description: "A PHP web framework for web artisans.",
    rating: 3,
    imageUrl: "/logos/laravel.svg",
    category: "Frameworks",
  },

  // Python Libraries
  {
    name: "Scikit-Learn",
    description: "The essential library for classical machine learning.",
    rating: 5,
    imageUrl: "/logos/scikit-learn.svg",
    category: "Python Libraries",
  },
  {
    name: "Pandas",
    description: "The fundamental library for data manipulation and analysis.",
    rating: 5,
    imageUrl: "/logos/pandas.svg",
    category: "Python Libraries",
  },
  {
    name: "NumPy",
    description: "The core library for numerical computing in Python.",
    rating: 5,
    imageUrl: "/logos/numpy.svg",
    category: "Python Libraries",
  },
  {
    name: "Matplotlib",
    description: "A comprehensive library for creating static and interactive plots.",
    rating: 4,
    imageUrl: "/logos/matplotlib.svg",
    category: "Python Libraries",
  },
  {
    name: "Seaborn",
    description: "High-level statistical data visualization based on Matplotlib.",
    rating: 4,
    imageUrl: "/logos/seaborn.svg",
    category: "Python Libraries",
  },
  {
    name: "OpenCV",
    description: "The leading library for computer vision and image processing.",
    rating: 4,
    imageUrl: "/logos/opencv.svg",
    category: "Python Libraries",
  },
  { // NEWLY ADDED
    name: "Plotly",
    description: "An interactive, open-source graphing library for Python.",
    rating: 4,
    imageUrl: "/logos/plotly.svg",
    category: "Python Libraries",
  },
  { // NEWLY ADDED
    name: "SciPy",
    description: "A core library for scientific and technical computing.",
    rating: 4,
    imageUrl: "/logos/scipy.svg",
    category: "Python Libraries",
  },

  // Database
  {
    name: "MySQL",
    description: "A reliable relational database for storing structured data.",
    rating: 4,
    imageUrl: "/logos/mysql.svg",
    category: "Database",
  },
  {
    name: "MongoDB",
    description: "A flexible NoSQL database for unstructured data.",
    rating: 4,
    imageUrl: "/logos/mongodb.svg",
    category: "Database",
  },

  // Cloud & MLOps
  {
    name: "Docker",
    description: "Containerization tool for reproducible environments.",
    rating: 4,
    imageUrl: "/logos/docker.svg",
    category: "Cloud & MLOps",
  },
  {
    name: "Hugging Face",
    description: "The leading platform for sharing and using ML models.",
    rating: 4,
    imageUrl: "/logos/huggingface.svg",
    category: "Cloud & MLOps",
  },
  {
    name: "GitHub Actions",
    description: "CI/CD pipelines for automating ML workflows.",
    rating: 4,
    imageUrl: "/logos/github-actions.svg",
    category: "Cloud & MLOps",
  },
  
  {
    name: "Git",
    description: "The industry-standard distributed version control system.",
    rating: 5,
    imageUrl: "/logos/git.svg",
    category: "Version Control",
  },
  {
    name: "GitHub",
    description: "A web-based platform for version control and collaboration.",
    rating: 5,
    imageUrl: "/logos/github.svg",
    category: "Version Control",
  },

  {
    name: "LaTeX",
    description: "A document preparation system for high-quality typesetting.",
    rating: 4,
    imageUrl: "/logos/latex.svg",
    category: "Office Skills",
  },
  // {
  //   name: "Google Sheets",
  //   description: "A web-based spreadsheet application.",
  //   rating: 4,
  //   imageUrl: "/logos/google-sheets.svg",
  //   category: "Office Skills",
  // },
  // {
  //   name: "Microsoft Word",
  //   description: "A leading word processing application.",
  //   rating: 4,
  //   imageUrl: "/logos/word.svg",
  //   category: "Office Skills",
  // },
  // {
  //   name: "PowerPoint",
  //   description: "A powerful presentation-making software.",
  //   rating: 4,
  //   imageUrl: "/logos/powerpoint.svg",
  //   category: "Office Skills",
  // },
];

export const skills = skillsUnsorted.slice().sort((a, b) => b.rating - a.rating);
export const featuredSkills = skills;



// import { Icons } from "@/components/common/icons";

// export interface skillsInterface {
//   name: string;
//   description: string;
//   rating: number;
//   icon: any;
// }

// export const skillsUnsorted: skillsInterface[] = [
//   {
//     name: "Next.js",
//     description:
//       "fuging build dynamic apps with routing, layouts, loading UI, and API routes.",
//     rating: 5,
//     icon: Icons.nextjs,
//   },
//   {
//     name: "React",
//     description:
//       "Craft interactive user interfaces using components, state, props, and virtual DOM.",
//     rating: 5,
//     icon: Icons.react,
//   },
//   {
//     name: "GraphQL",
//     description:
//       "Fetch data precisely with a powerful query language for APIs and runtime execution.",
//     rating: 4,
//     icon: Icons.graphql,
//   },
//   {
//     name: "Nest.js",
//     description:
//       "Create scalable and modular applications with a progressive Node.js framework.",
//     rating: 4,
//     icon: Icons.nestjs,
//   },
//   {
//     name: "express.js",
//     description:
//       "Build web applications and APIs quickly using a fast, unopinionated Node.js framework.",
//     rating: 5,
//     icon: Icons.express,
//   },
//   {
//     name: "Node.js",
//     description:
//       "Run JavaScript on the server side, enabling dynamic and responsive applications.",
//     rating: 5,
//     icon: Icons.nodejs,
//   },
//   {
//     name: "MongoDB",
//     description:
//       "Store and retrieve data seamlessly with a flexible and scalable NoSQL database.",
//     rating: 5,
//     icon: Icons.mongodb,
//   },
//   {
//     name: "Typescript",
//     description:
//       "Enhance JavaScript with static types, making code more understandable and reliable.",
//     rating: 5,
//     icon: Icons.typescript,
//   },
//   {
//     name: "Javascript",
//     description:
//       "Create interactive and dynamic web experiences with the versatile scripting language.",
//     rating: 5,
//     icon: Icons.javascript,
//   },
//   {
//     name: "HTML 5",
//     description:
//       "Structure web content beautifully with the latest version of HyperText Markup Language.",
//     rating: 4,
//     icon: Icons.html5,
//   },
//   {
//     name: "CSS 3",
//     description:
//       "Style web pages creatively with the latest iteration of Cascading Style Sheets.",
//     rating: 4,
//     icon: Icons.css3,
//   },
//   {
//     name: "React Native",
//     description:
//       "Develop cross-platform mobile apps using React for consistent and engaging experiences.",
//     rating: 4,
//     icon: Icons.react,
//   },
//   {
//     name: "Angular",
//     description:
//       "Build dynamic web apps with a TypeScript-based open-source framework by Google.",
//     rating: 3,
//     icon: Icons.angular,
//   },
//   {
//     name: "Redux",
//     description:
//       "Manage app state effectively using a predictable and centralized state container.",
//     rating: 4,
//     icon: Icons.redux,
//   },
//   {
//     name: "Socket.io",
//     description:
//       "Enable real-time, bidirectional communication between clients and servers effortlessly.",
//     rating: 3,
//     icon: Icons.socketio,
//   },
//   {
//     name: "Material UI",
//     description:
//       "Create stunning and responsive UIs with a popular React UI framework.",
//     rating: 4,
//     icon: Icons.mui,
//   },

//   {
//     name: "Tailwind CSS",
//     description:
//       "Design beautiful, modern websites faster with a utility-first CSS framework.",
//     rating: 5,
//     icon: Icons.tailwindcss,
//   },
//   {
//     name: "AWS",
//     description:
//       "Utilize Amazon Web Services to build and deploy scalable, reliable, and secure applications.",
//     rating: 3,
//     icon: Icons.amazonaws,
//   },
//   {
//     name: "Bootstrap",
//     description:
//       "Quickly create responsive and appealing web designs using a popular CSS framework.",
//     rating: 2,
//     icon: Icons.bootstrap,
//   },
//   {
//     name: "MySQL",
//     description:
//       "Manage and organize relational databases efficiently for data-driven applications.",
//     rating: 2,
//     icon: Icons.mysql,
//   },
//   {
//     name: "Netlify",
//     description:
//       "Manage and organize relational databases efficiently for data-driven applications.",
//     rating: 4,
//     icon: Icons.netlify,
//   },
// ];

// export const skills = skillsUnsorted
//   .slice()
//   .sort((a, b) => b.rating - a.rating);

// export const featuredSkills = skills.slice(0, 6);
