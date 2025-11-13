/* This is the new, correct app/(root)/education/[expId]/page.tsx */

import { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { redirect } from "next/navigation";

import { AnimatedSection } from "@/components/common/animated-section";
import { ClientPageWrapper } from "@/components/common/client-page-wrapper";
import { Icons } from "@/components/common/icons";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { ResponsiveTabs } from "@/components/ui/responsive-tabs";
// 1. IMPORT 'education' DATA
import { education } from "@/config/education";
import { siteConfig } from "@/config/site";

// 2. RENAME PROPS INTERFACE
interface EducationDetailPageProps {
  params: {
    expId: string;
  };
}

// Helper function to extract year from date
const getYearFromDate = (date: Date): string => {
  return new Date(date).getFullYear().toString();
};

// 3. UPDATE HELPER FUNCTION TO HANDLE 'null'
const getDurationText = (
  startDate: Date,
  endDate: Date | "Present" | null // Can be null
): string => {
  const startYear = getYearFromDate(startDate);
  const endYear =
    typeof endDate === "string"
      ? endDate
      : endDate === null // Check for null
      ? "Present"
      : getYearFromDate(endDate);
  return `${startYear} - ${endYear}`;
};

export async function generateMetadata({
  params,
}: EducationDetailPageProps): Promise<Metadata> { // 4. RENAME PROP TYPE
  // 5. FIND IN 'education' ARRAY
  const item = education.find((c) => c.id === params.expId);

  if (!item) {
    return {
      title: "Education Not Found", // 6. UPDATE "NOT FOUND"
    };
  }

  return {
    // 7. UPDATE METADATA
    title: `${item.position} | Education`,
    description: `Details about my ${item.position} at ${item.company}.`,
    alternates: {
      canonical: `${siteConfig.url}/education/${params.expId}`,
    },
  };
}

// 8. RENAME FUNCTION
export default function EducationDetailPage({
  params,
}: EducationDetailPageProps) { // 9. RENAME PROP TYPE
  // 10. FIND IN 'education' ARRAY
  const item = education.find((c) => c.id === params.expId);

  if (!item) {
    // 11. REDIRECT TO 'education'
    redirect("/education");
  }

  const tabItems = [
    {
      value: "summary",
      label: "Summary",
      content: (
        <AnimatedSection delay={0.3}>
          <div>
            <h3 className="font-semibold mb-4 text-sm uppercase tracking-wide text-muted-foreground">
              Summary
            </h3>
            <ul className="space-y-3">
              {/* 12. USE 'item.description' */}
              {item.description.map((desc, idx) => (
                <li
                  key={idx}
                  className="text-base leading-relaxed flex items-start gap-3"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                  {desc}
                </li>
              ))}
            </ul>
          </div>
        </AnimatedSection>
      ),
    },
    {
      // 13. RENAME "Achievements" TAB
      value: "details",
      label: "Details & Courses",
      content: (
        <AnimatedSection delay={0.3}>
          <div>
            <h3 className="font-semibold mb-4 text-sm uppercase tracking-wide text-muted-foreground">
              Key Courses & Details
            </h3>
            <ul className="space-y-3">
              {/* 14. USE 'item.descriptionDetails' */}
              {item.descriptionDetails.map((detail, idx) => (
                <li
                  key={idx}
                  className="text-base leading-relaxed flex items-start gap-3"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                  {detail}
                </li>
              ))}
            </ul>
          </div>
        </AnimatedSection>
      ),
    },
    // 15. "Skills" TAB IS DELETED
  ];

  return (
    <ClientPageWrapper>
      <div className="container max-w-4xl mx-auto py-8 px-4">
        <AnimatedSection className="mb-6">
          <Button variant="ghost" size="sm" className="mb-4" asChild>
            {/* 16. UPDATE "BACK" LINK */}
            <Link href="/education">
              <Icons.chevronLeft className="mr-2 h-4 w-4" />
              Back to Education
            </Link>
          </Button>
        </AnimatedSection>

        <AnimatedSection delay={0.2}>
          <Card className="overflow-hidden rounded-lg border bg-background p-2 transition-all duration-300">
            <CardHeader className="pb-6">
              <div className="flex flex-col gap-4">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                  <div className="flex flex-col sm:flex-row items-center sm:items-start gap-4">
                    {/* 17. USE 'item.' FOR ALL PROPERTIES */}
                    {item.logo && (
                      <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-lg border-2 border-border overflow-hidden bg-white flex-shrink-0">
                        <Image
                          src={item.logo}
                          alt={item.company}
                          width={80}
                          height={80}
                          className="w-full h-full object-contain p-2"
                        />
                      </div>
                    )}
                    <div className="flex-1 text-center sm:text-left">
                      <h1 className="text-xl sm:text-2xl md:text-3xl font-bold mb-2">
                        {item.position}
                      </h1>
                      <div className="flex items-center justify-center sm:justify-start gap-2 mb-2">
                        <span className="text-md font-medium text-muted-foreground">
                          {item.company}
                        </span>
                        {item.companyUrl && (
                          <a
                            href={item.companyUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-muted-foreground hover:text-foreground transition-colors"
                          >
                            <Icons.externalLink className="w-4 h-4" />
                          </a>
                        )}
                      </div>
                      <p className="text-muted-foreground">
                        {item.location}
                      </p>
                    </div>
                  </div>
                  <div className="flex justify-center sm:justify-end">
                    <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-primary/10 text-primary border border-primary/20">
                      {getDurationText(
                        item.startDate,
                        item.endDate
                      )}
                    </span>
                  </div>
                </div>
              </div>
            </CardHeader>

            <CardContent>
              <ResponsiveTabs items={tabItems} defaultValue="summary" />
            </CardContent>
          </Card>
        </AnimatedSection>

        <AnimatedSection delay={0.4} className="flex justify-center mt-8">
          <Button variant="outline" asChild>
            {/* 18. UPDATE "VIEW ALL" LINK */}
            <Link href="/education">
              <Icons.chevronLeft className="mr-2 h-4 w-4" />
              View All Education
            </Link>
          </Button>
        </AnimatedSection>
      </div>
    </ClientPageWrapper>
  );
}