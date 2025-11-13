// THIS IS YOUR NEW config/education.ts

// This is the correct, full interface.
// It includes `descriptionDetails` for your course list.
export interface EducationInterface {
  id: string;
  company: string;
  position: string;
  logo: string;
  location: string;
  companyUrl: string | null;
  startDate: Date;
  endDate: Date | null;
  description: string[];
  descriptionDetails: string[]; // <-- THIS IS FOR YOUR DETAILS PAGE
  skills: string[];
}

export const education: EducationInterface[] = [
  {
    id: "edu-bsc",
    company: "BRAC University",
    position: "BSc in Computer Science & Engineering",
    logo: "/logos/bracu.svg",
    location: "Dhaka,     Bangladesh",
    companyUrl: "https://www.bracu.ac.bd/",
    startDate: new Date("2020-01-01"),
    endDate: new Date("2025-01-01"),
    description: ["CGPA: 3.17 out of 4.00"],
    descriptionDetails: [ // <-- ADD YOUR COURSES HERE
      "Relevant Courses:",
      "- Data Structures and Algorithms",
      "- Artificial Intelligence",
      "- Machine Learning",
      "- Database Systems",
      "- Software Engineering",
    ],
    skills: [],
  },
  {
    id: "edu-hsc",
    company: "Abu Abbas College",
    position: "Higher Secondary Certificate (HSC)",
    logo: "/logos/college.svg",
    location: "Netrakona, Bangladesh",
    companyUrl: null,
    startDate: new Date("2017-01-01"),
    endDate: new Date("2019-01-01"),
    description: ["GPA: 4.50 out of 5.00"],
    descriptionDetails: [
      "Completed the Higher Secondary Certificate in the Science group.",
    ],
    skills: [],
  },
  {
    id: "edu-ssc",
    company: "Anjuman Adarsha Govt High School",
    position: "Secondary School Certificate (SSC)",
    logo: "/logos/school.svg",
    location: "Netrakona, Bangladesh",
    companyUrl: null,
    startDate: new Date("2015-01-01"),
    endDate: new Date("2017-01-01"),
    description: ["GPA: 5.00 out of 5.00"],
    descriptionDetails: [
      "Completed the Secondary School Certificate in the Science group.",
    ],
    skills: [],
  },
];










// /*
//   This is the new, correct config/education.ts file.
//   We are "tricking" the ExperienceCard by giving it an object
//   that has the *exact same shape* as an Experience object.
// */

// // We must use the *exact* interface from config/experience.ts
// // or a compatible one. This one is compatible.
// export interface EducationInterface {
//   id: string;
//   company: string;     // Renamed from companyName
//   position: string;    // Renamed from role
//   logo: string;        // Renamed from companyLogoImg
//   location: string;    // Added this field
//   companyUrl: string | null; // Added this field
//   startDate: Date;
//   endDate: Date | null;
//   description: string[]; // Changed to an array of strings
//   skills: string[];
// }

// export const education: EducationInterface[] = [
//   {
//     id: "edu-bsc",
//     company: "BRAC University",
//     position: "BSc in Computer Science & Engineering",
//     logo: "/logos/bracu.svg",
//     location: "Dhaka, Bangladesh",
//     companyUrl: null,
//     startDate: new Date("2020-01-01"),
//     endDate: new Date("2025-01-01"),
//     description: ["CGPA: 3.17 out of 4.00"], // <-- Now an array
//     skills: [], // <-- This prevents the crash
//   },
//   {
//     id: "edu-hsc",
//     company: "Abu Abbas College",
//     position: "Higher Secondary Certificate (HSC)",
//     logo: "/logos/college.svg",
//     location: "Netrakona, Bangladesh",
//     companyUrl: null,
//     startDate: new Date("2017-01-01"),
//     endDate: new Date("2019-01-01"),
//     description: ["GPA: 5.00 out of 5.00"], // <-- Now an array
//     skills: [], // <-- This prevents the crash
//   },
//   {
//     id: "edu-ssc",
//     company: "Anjuman Adarsha Govt High School",
//     position: "Secondary School Certificate (SSC)",
//     logo: "/logos/school.svg",
//     location: "Netrakona, Bangladesh",
//     companyUrl: null,
//     startDate: new Date("2015-01-01"),
//     endDate: new Date("2017-01-01"),
//     description: ["GPA: 5.00 out of 5.00"], // <-- Now an array
//     skills: [], // <-- This prevents the crash
//   },
// ];