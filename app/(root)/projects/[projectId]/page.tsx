// app/projects/[id]/page.tsx
import Image from "next/image";
import Link from "next/link";
import { redirect } from "next/navigation";
import React from "react";
import ReactMarkdown from 'react-markdown';
import { Icons } from "@/components/common/icons";
import ProjectDescription from "@/components/projects/project-description";
import { buttonVariants } from "@/components/ui/button";
import ChipContainer from "@/components/ui/chip-container";
import CustomTooltip from "@/components/ui/custom-tooltip";
import { Projects } from "@/config/projects";
import { siteConfig } from "@/config/site";
import { cn, formatDateFromObj } from "@/lib/utils";
import profileImg from "@/public/profile-img.jpg";

interface ProjectPageProps {
  params: {
    projectId: string;
  };
}

export default function Project({ params }: ProjectPageProps) {
  let project = Projects.find((val) => val.id === params.projectId);
  if (!project) {
    redirect("/projects");
  }

  return (
    <article className="container relative max-w-3xl py-6 lg:py-10">
      <Link
        href="/projects"
        className={cn(
          buttonVariants({ variant: "ghost" }),
          "absolute left-[-200px] top-14 hidden xl:inline-flex"
        )}
      >
        <Icons.chevronLeft className="mr-2 h-4 w-4" />
        All Projects
      </Link>

      <div>
        <time
          dateTime={Date.now().toString()}
          className="block text-sm text-muted-foreground"
        >
          {project.startDate ? formatDateFromObj(project.startDate) : null}
        </time>

        <h1 className="flex items-center justify-between mt-2 font-heading text-4xl leading-tight lg:text-5xl">
          {project.companyName}
          <div className="flex items-center">
            {project.githubLink && (
              <CustomTooltip text="Link to the source code.">
                <Link href={project.githubLink} target="_blank">
                  <Icons.gitHub className="w-6 ml-4 text-muted-foreground hover:text-foreground" />
                </Link>
              </CustomTooltip>
            )}
            {project.websiteLink && (
              <CustomTooltip text="Please note that some project links may be temporarily unavailable.">
                <Link href={project.websiteLink} target="_blank">
                  <Icons.externalLink className="w-6 ml-4 text-muted-foreground hover:text-foreground " />
                </Link>
              </CustomTooltip>
            )}
          </div>
        </h1>

        <ChipContainer textArr={project.category} />

        <div className="mt-4 flex space-x-4">
          <Link href={siteConfig.links.github} className="flex items-center space-x-2 text-sm">
            <Image src={profileImg} alt={"alve"} width={42} height={42} className="rounded-full bg-background" />

            <div className="flex-1 text-left leading-tight">
              <p className="font-medium">{"Azmain Khan Alve"}</p>
              <p className="text-[12px] text-muted-foreground">@{siteConfig.username}</p>
            </div>
          </Link>
        </div>
      </div>

      {/* HERO IMAGE / METRICS */}
      <div className="my-8">
        <Image
          src={project.companyLogoImg}
          alt={project.companyName}
          width={720}
          height={405}
          className="rounded-md border bg-muted transition-colors w-full object-cover"
          priority
        />
      </div>

      {/* HERO METRICS (if available) */}
      {project.heroInfo && (
        <section className="mb-6">
          <h2 className="font-heading text-2xl mb-2">{project.heroInfo.headline}</h2>
          {project.heroInfo.keyMetrics && (
            <div className="flex flex-wrap gap-6 mt-3">
              {project.heroInfo.keyMetrics.map((m, i) => (
                <div key={i} className="p-3 bg-muted rounded-md text-center min-w-[140px]">
                  <div className="text-xl font-semibold">{m.value}</div>
                  <div className="text-xs text-muted-foreground">{m.label}</div>
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      <div className="mb-7 ">
        <h2 className="inline-block font-heading text-3xl leading-tight lg:text-3xl mb-2">Tech Stack</h2>
        <ChipContainer textArr={project.techStack} />
      </div>

      {/* DESCRIPTION: Prefer new sections if present, otherwise fallback to old component */}
      <div className="mb-7 ">
        <h2 className="inline-block font-heading text-3xl leading-tight lg:text-3xl mb-2">Description</h2>

        {project.descriptionSections && project.descriptionSections.length > 0 ? (
          <div className="space-y-6">
            {project.descriptionSections.map((sec, idx) => (
              <section key={idx} className="prose max-w-none prose-ul:pl-6">
                <h3 className="text-xl font-semibold">{sec.title}</h3>
                {sec.content.map((p, j) => (
                  <ReactMarkdown key={j}>{p}</ReactMarkdown>
                ))}
              </section>
            ))}
          </div>
        ) : (
          <ProjectDescription
            paragraphs={project.descriptionDetails?.paragraphs || []}
            bullets={project.descriptionDetails?.bullets || []}
          />
          // <ProjectDescription paragraphs={project.descriptionDetails.paragraphs} bullets={project.descriptionDetails.bullets} />
        )}
      </div>

      {/* PAGE INFO (images + small sections) */}
      <div className="mb-7 ">
        <h2 className="inline-block font-heading text-3xl leading-tight lg:text-3xl mb-5">Page Info</h2>
        {project.pagesInfoArr.map((page, ind) => (
          <div key={ind}>
            <h3 className="flex items-center font-heading text-xl leading-tight lg:text-xl mt-3">
              <Icons.star className="h-5 w-5 mr-2" /> {page.title}
            </h3>
            <div>
              {page.description && <ReactMarkdown>{page.description}</ReactMarkdown>}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 my-4">
                {page.imgArr.map((img, i) => (
                  <Image
                    src={img}
                    key={i}
                    alt={`${page.title} image ${i + 1}`}
                    width={720}
                    height={405}
                    className="rounded-md border bg-muted transition-colors object-cover"
                    priority
                  />
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Technical Appendix (collapsible) */}
      {project.technicalAppendix && (
        <details className="mb-8 border rounded-md p-4">
          <summary className="cursor-pointer font-semibold">{project.technicalAppendix.title}</summary>
          <div className="mt-3 space-y-2 text-gray-700">
            {project.technicalAppendix.content.map((line, i) => (
              <ReactMarkdown key={i}>{line}</ReactMarkdown>
            ))}
          </div>
        </details>
      )}

      <hr className="mt-12" />
      <div className="flex justify-center py-6 lg:py-10">
        <Link href="/projects" className={cn(buttonVariants({ variant: "ghost" }))}>
          <Icons.chevronLeft className="mr-2 h-4 w-4" />
          All Projects
        </Link>
      </div>
    </article>
  );
}
















// ============================================================





// import Image from "next/image";
// import Link from "next/link";
// import { redirect } from "next/navigation";

// import { Icons } from "@/components/common/icons";
// import ProjectDescription from "@/components/projects/project-description";
// import { buttonVariants } from "@/components/ui/button";
// import ChipContainer from "@/components/ui/chip-container";
// import CustomTooltip from "@/components/ui/custom-tooltip";
// import { Projects } from "@/config/projects";
// import { siteConfig } from "@/config/site";
// import { cn, formatDateFromObj } from "@/lib/utils";
// import profileImg from "@/public/profile-img.jpg";

// interface ProjectPageProps {
//   params: {
//     projectId: string;
//   };
// }


// export default function Project({ params }: ProjectPageProps) {
//   let project = Projects.find((val) => val.id === params.projectId);
//   if (!project) {
//     redirect("/projects");
//   }

//   return (
//     <article className="container relative max-w-3xl py-6 lg:py-10">
//       <Link
//         href="/projects"
//         className={cn(
//           buttonVariants({ variant: "ghost" }),
//           "absolute left-[-200px] top-14 hidden xl:inline-flex"
//         )}
//       >
//         <Icons.chevronLeft className="mr-2 h-4 w-4" />
//         All Projects
//       </Link>
//       <div>
//         <time
//           dateTime={Date.now().toString()}
//           className="block text-sm text-muted-foreground"
//         >
//           {/* {formatDateFromObj(project.startDate)} */}
//           {project.startDate ? formatDateFromObj(project.startDate) : null}
//         </time>
//         <h1 className="flex items-center justify-between mt-2 font-heading text-4xl leading-tight lg:text-5xl">
//           {project.companyName}
//           <div className="flex items-center">
//             {project.githubLink && (
//               <CustomTooltip text="Link to the source code.">
//                 <Link href={project.githubLink} target="_blank">
//                   <Icons.gitHub className="w-6 ml-4 text-muted-foreground hover:text-foreground" />
//                 </Link>
//               </CustomTooltip>
//             )}
//             {project.websiteLink && (
//               <CustomTooltip text="Please note that some project links may be temporarily unavailable.">
//                 <Link href={project.websiteLink} target="_blank">
//                   <Icons.externalLink className="w-6 ml-4 text-muted-foreground hover:text-foreground " />
//                 </Link>
//               </CustomTooltip>
//             )}
//           </div>
//         </h1>
//         <ChipContainer textArr={project.category} />
//         <div className="mt-4 flex space-x-4">
//           <Link
//             href={siteConfig.links.github}
//             className="flex items-center space-x-2 text-sm"
//           >
//             <Image
//               src={profileImg}
//               alt={"alve"}
//               width={42}
//               height={42}
//               className="rounded-full bg-background"
//             />

//             <div className="flex-1 text-left leading-tight">
//               <p className="font-medium">{"Azmain Khan Alve"}</p>
//               <p className="text-[12px] text-muted-foreground">
//                 @{siteConfig.username}
//               </p>
//             </div>
//           </Link>
//         </div>
//       </div>

//       <Image
//         src={project.companyLogoImg}
//         alt={project.companyName}
//         width={720}
//         height={405}
//         className="my-8 rounded-md border bg-muted transition-colors"
//         priority
//       />

//       <div className="mb-7 ">
//         <h2 className="inline-block font-heading text-3xl leading-tight lg:text-3xl mb-2">
//           Tech Stack
//         </h2>
//         <ChipContainer textArr={project.techStack} />
//       </div>

//       <div className="mb-7 ">
//         <h2 className="inline-block font-heading text-3xl leading-tight lg:text-3xl mb-2">
//           Description
//         </h2>
//         {/* {<project.descriptionComponent />} */}
//         <ProjectDescription
//           paragraphs={project.descriptionDetails.paragraphs}
//           bullets={project.descriptionDetails.bullets}
//         />
//       </div>

//       <div className="mb-7 ">
//         <h2 className="inline-block font-heading text-3xl leading-tight lg:text-3xl mb-5">
//           Page Info
//         </h2>
//         {project.pagesInfoArr.map((page, ind) => (
//           <div key={ind}>
//             <h3 className="flex items-center font-heading text-xl leading-tight lg:text-xl mt-3">
//               <Icons.star className="h-5 w-5 mr-2" /> {page.title}
//             </h3>
//             <div>
//               <p>{page.description}</p>
//               {page.imgArr.map((img, ind) => (
//                 <Image
//                   src={img}
//                   key={ind}
//                   alt={img}
//                   width={720}
//                   height={405}
//                   className="my-4 rounded-md border bg-muted transition-colors"
//                   priority
//                 />
//               ))}
//             </div>
//           </div>
//         ))}
//       </div>

//       <hr className="mt-12" />
//       <div className="flex justify-center py-6 lg:py-10">
//         <Link
//           href="/projects"
//           className={cn(buttonVariants({ variant: "ghost" }))}
//         >
//           <Icons.chevronLeft className="mr-2 h-4 w-4" />
//           All Projects
//         </Link>
//       </div>
//     </article>
//   );
// }
