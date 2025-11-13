"use client"; // We still need this here

import Image from "next/image"; // We must import the Image component
import { skillsInterface } from "@/config/skills";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardDescription,
} from "@/components/ui/card";

export default function SkillsCard({ skills }: { skills: skillsInterface[] }) {

  const categoryOrder: string[] = [
    "Languages",
    "Frameworks",
    "Python Libraries",
    "Database",
    "Cloud & MLOps",
    "Version Control",
    "Office Skills",
  ];

  const groupedSkills = skills.reduce(
    (acc, skill) => {
      const category = skill.category || "Other";
      if (!acc[category]) {
        acc[category] = [];
      }
      acc[category].push(skill);
      return acc;
    },
    {} as Record<string, skillsInterface[]>
  );

  return (
    <div className="space-y-12">
      {categoryOrder.map((category) =>
        groupedSkills[category] ? (
          <section key={category}>
            <h2 className="font-heading text-2xl sm:text-3xl md:text-4xl mb-6 text-left">
              {category}
            </h2>
            <div className="mx-auto grid justify-center gap-4 md:w-full lg:grid-cols-3">
              {groupedSkills[category].map((skill) => (
                <Card
                  key={skill.name}
                  className="flex flex-col justify-between"
                >
                  <CardHeader>
                    <div className="flex items-center gap-4"> {/* Increased gap for looks */}
                      {/* THIS IS THE NEW CODE BLOCK */}
                      <Image
                        src={skill.imageUrl} // Using the image path from your config
                        alt={`${skill.name} logo`}
                        width={32}  // Setting a fixed width
                        height={32} // Setting a fixed height
                        className="h-8 w-8" // Tailwind classes to control size
                      />
                      <CardTitle className="text-lg">{skill.name}</CardTitle>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <CardDescription>{skill.description}</CardDescription>
                  </CardContent>
                </Card>
              ))}
            </div>
          </section>
        ) : null
      )}
    </div>
  );
}








// "use client";
// import { skillsInterface } from "@/config/skills";
// import {
//   Card,
//   CardHeader,
//   CardTitle,
//   CardContent,
//   CardDescription,
// } from "@/components/ui/card";

// // This is the new component code.
// // It is smart: it groups your skills by category BEFORE rendering them.
// export default function SkillsCard({ skills }: { skills: skillsInterface[] }) {

//   // 1. Define the order of your categories
//   const categoryOrder: string[] = [
//     "Languages",
//     "Frameworks",
//     "Python Libraries",
//     "Database",
//     "Cloud & MLOps",
//   ];

//   // 2. Group the skills by category
//   const groupedSkills = skills.reduce(
//     (acc, skill) => {
//       const category = skill.category || "Other";
//       if (!acc[category]) {
//         acc[category] = [];
//       }
//       acc[category].push(skill);
//       return acc;
//     },
//     {} as Record<string, skillsInterface[]>
//   );

//   // 3. Render the categories IN THE ORDER YOU DEFINED
//   return (
//     <div className="space-y-12">
//       {categoryOrder.map((category) =>
//         // Only render the section if that category exists in the list
//         groupedSkills[category] ? (
//           <section key={category}>
//             <h2 className="font-heading text-2xl sm:text-3xl md:text-4xl mb-6 text-left">
//               {category}
//             </h2>
//             <div className="mx-auto grid justify-center gap-4 md:w-full lg:grid-cols-3">
//               {groupedSkills[category].map((skill) => (
//                 <Card
//                   key={skill.name}
//                   className="flex flex-col justify-between"
//                 >
//                   <CardHeader>
//                     <div className="flex items-center gap-3">
//                       <span className="flex h-8 w-8 items-center justify-center">
//                         {skill.icon}
//                       </span>
//                       <CardTitle className="text-lg">{skill.name}</CardTitle>
//                     </div>
//                   </CardHeader>
//                   <CardContent>
//                     <CardDescription>{skill.description}</CardDescription>
//                   </CardContent>
//                 </Card>
//               ))}
//             </div>
//           </section>
//         ) : null
//       )}
//     </div>
//   );
// }










// import Rating from "@/components/skills/rating";
// import { skillsInterface } from "@/config/skills";

// interface SkillsCardProps {
//   skills: skillsInterface[];
// }

// export default function SkillsCard({ skills }: SkillsCardProps) {
//   return (
//     <div className="mx-auto grid justify-center gap-4 sm:grid-cols-2 lg:grid-cols-3">
//       {skills.map((skill, id) => (
//         <div
//           key={id}
//           className="relative overflow-hidden rounded-lg border bg-background p-2"
//         >
//           <div className="flex h-[230px] flex-col justify-between rounded-md p-6 sm:h-[230px]">
//             <skill.icon size={50} />
//             <div className="space-y-2">
//               <h3 className="font-bold">{skill.name}</h3>
//               <p className="text-sm text-muted-foreground">
//                 {skill.description}
//               </p>
//               <Rating stars={skill.rating} />
//             </div>
//           </div>
//         </div>
//       ))}
//     </div>
//   );
// }
