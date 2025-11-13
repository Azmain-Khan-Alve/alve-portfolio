
//===================(newly added)
"use client";
//====================

//=========(new commented out)
// import { Metadata } from "next";

import PageContainer from "@/components/common/page-container";
import SkillsCard from "@/components/skills/skills-card";
import { pagesConfig } from "@/config/pages";
import { skills } from "@/config/skills";

//=========(new commented out)
// export const metadata: Metadata = {
//   title: pagesConfig.skills.metadata.title,
//   description: pagesConfig.skills.metadata.description,
// };

export default function SkillsPage() {
  return (
    <PageContainer
      title={pagesConfig.skills.title}
      description={pagesConfig.skills.description}
    >
      <SkillsCard skills={skills} />
    </PageContainer>
  );
}
