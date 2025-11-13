"use client";

import Image from "next/image";
import Link from "next/link";
import React from "react";

import { Icons } from "@/components/common/icons";
import { Button } from "@/components/ui/button";
// 1. Import your EducationInterface
import { EducationInterface } from "@/config/education";

// Helper function to extract year from date
const getYearFromDate = (date: Date): string => {
  return new Date(date).getFullYear().toString();
};

// 2. Update helper function to handle 'null' for "Present"
const getDurationText = (
  startDate: Date,
  endDate: Date | "Present" | null // Now handles null
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

// 3. Rename the props interface
interface EducationCardProps {
  education: EducationInterface; // Use EducationInterface
}

// 4. Rename the component
const EducationCard: React.FC<EducationCardProps> = ({ education }) => {
  return (
    <div className="group relative overflow-hidden rounded-lg border bg-background p-4 sm:p-6 transition-all duration-300">
      <div className="flex items-start gap-3 sm:gap-4">
        {/* 5. Use 'education' prop */}
        {education.logo && (
          <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-lg border-2 border-border overflow-hidden bg-white flex-shrink-0">
            <Image
              src={education.logo}
              alt={education.company}
              width={48}
              height={48}
              className="w-full h-full object-contain p-2"
            />
          </div>
        )}
        <div className="flex-1 min-w-0">
          <div className="flex flex-col gap-1 sm:gap-2">
            <div className="flex items-start sm:items-center gap-2">
              <h3 className="text-base sm:text-lg font-bold text-foreground line-clamp-2 sm:line-clamp-1">
                {education.position}
              </h3>
              {education.companyUrl && (
                <a
                  href={education.companyUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted-foreground hover:text-foreground transition-colors flex-shrink-0 mt-0.5 sm:mt-0"
                >
                  <Icons.externalLink className="w-4 h-4" />
                </a>
              )}
            </div>
            <div className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-2 text-sm text-muted-foreground">
              <span className="font-medium">{education.company}</span>
              <span className="hidden sm:inline">•</span>
              <span>{education.location}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-primary/10 text-primary border border-primary/20">
                {getDurationText(education.startDate, education.endDate)}
              </span>
            </div>
          </div>
          <p className="mt-2 sm:mt-3 text-sm text-muted-foreground line-clamp-2">
            {education.description[0]}
          </p>
          {/* This part is fine, it will just show nothing since skills: [] */}
          <div className="mt-3 sm:mt-4 flex flex-wrap gap-1">
            {education.skills.slice(0, 2).map((skill, index) => (
              <span
                key={index}
                className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-muted text-muted-foreground"
              >
                {skill}
              </span>
            ))}
            {education.skills.length > 2 && (
              <span className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-muted text-muted-foreground">
                +{education.skills.length - 2} more
              </span>
            )}
          </div>
        </div>
      </div>
      <div className="mt-3 sm:mt-4 flex justify-end">
        <Button
          variant="outline"
          size="sm"
          className="rounded-lg w-full sm:w-auto"
          asChild
        >
          {/* 6. This is the critical fix: changed href to /education/ */}
          <Link href={`/education/${education.id}`}>
            View Details
            <Icons.chevronRight className="ml-2 h-4 w-4" />
          </Link>
        </Button>
      </div>
    </div>
  );
};

export default EducationCard;