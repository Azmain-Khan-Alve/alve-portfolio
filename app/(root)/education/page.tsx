/* This is the new, correct app/(root)/education/page.tsx */

import { Metadata } from "next";
import PageContainer from "@/components/common/page-container";
import { AnimatedSection } from "@/components/common/animated-section";
// 1. THIS IS THE MISSING IMPORT
import { education } from "@/config/education";
import { pagesConfig } from "@/config/pages";
import { siteConfig } from "@/config/site";
// 2. We import your new EducationCard
import EducationCard from "@/components/education/education-card";

export const metadata: Metadata = {
  title: `${pagesConfig.education.metadata.title} | Academic Background`,
  description: `${pagesConfig.education.metadata.description} My academic background and qualifications.`,
  keywords: [
    "education",
    "academic",
    "computer science",
    "bachelor of science",
    "brac university",
  ],
  alternates: {
    canonical: `${siteConfig.url}/education`,
  },
};

export default function EducationPage() {
  return (
    <PageContainer
      title={pagesConfig.education.title}
      description={pagesConfig.education.description}
    >
      {/* We create a grid, just like the projects page */}
      <div className="mx-auto grid justify-center gap-4 md:w-full lg:grid-cols-3">
        {/* 3. This line will now work because 'education' is imported */}
        {education.map((edu, index) => (
          <AnimatedSection
            key={edu.id}
            delay={0.1 * (index + 1)}
            direction="up"
          >
            {/* 4. We use the new <EducationCard> and pass the 'education' prop */}
            <EducationCard education={edu} />
          </AnimatedSection>
        ))}
      </div>
    </PageContainer>
  );
}