import { ValidPages } from "./constants";

type PagesConfig = {
  [key in ValidPages]: {
    title: string;
    description: string;
    metadata: {
      title: string;
      description: string;
    };
    // featuredDescription: string;
  };
};

export const pagesConfig: PagesConfig = {
  home: {
    title: "Home",
    description: "Welcome to my portfolio website.",
    metadata: {
      title: "Home",
      description: "Azmain Khan Alve's portfolio website.",
    },
  },
  skills: {
    title: "Skills",
    description: "Key skills that define my professional identity.",
    metadata: {
      title: "Skills",
      description:
        "Azmain Khan Alve's key skills that define his professional identity.",
    },
  },
  projects: {
    title: "Projects",
    description: "Showcasing impactful projects and technical achievements.",
    metadata: {
      title: "Projects",
      description: "Azmain Khan Alve's projects in building AI/ML applications.",
    },
  },
  contact: {
    title: "Contact",
    description: "Let's connect and explore collaborations.",
    metadata: {
      title: "Contact",
      description: "Contact Azmain Khan Alve.",
    },
  },
  // contributions: {
  //   title: "Contributions",
  //   description: "Open-source contributions and community involvement.",
  //   metadata: {
  //     title: "Contributions",
  //     description:
  //       "Azmain Khan Alve's open-source contributions and community involvement.",
  //   },
  // },
  resume: {
    title: "Resume",
    description: "Azmain Khan Alve's resume.",
    metadata: {
      title: "Resume",
      description: "Azmain Khan Alve's resume.",
    },
  },
  // experience: {
  //   title: "Experience",
  //   description: "Professional journey and career timeline.",
  //   metadata: {
  //     title: "Experience",
  //     description:
  //       "Azmain Khan Alve's professional journey and experience timeline.",
  //   },
  // },
  education: {
    title: "Education",
    description: "My academic background and qualifications.",
    metadata: {
      title: "Education",
      description:
        "Azmain Khan Alve's academic background and qualifications.",
    },
  },
};
