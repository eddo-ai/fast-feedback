import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { SampleCard } from "../SampleCard";

describe("SampleCard", () => {
  it("renders tags", () => {
    render(
      <SampleCard
        sample_id="1"
        pseudo_student_id="A1"
        text="Example text"
        tags={["Exemplar", "Misconception"]}
      />
    );
    expect(screen.getByText("Exemplar")).toBeInTheDocument();
    expect(screen.getByText("Misconception")).toBeInTheDocument();
  });
});
