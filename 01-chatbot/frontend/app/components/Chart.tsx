"use client";

import { useEffect, useRef } from "react";
import * as d3 from "d3";

export interface ChartData {
  type: "line" | "scatter" | "histogram" | "bar";
  title?: string;
  xLabel?: string;
  yLabel?: string;
  data: number[] | { x: number; y: number }[] | { label: string; value: number }[];
}

interface ChartProps {
  chartData: ChartData;
  darkMode?: boolean;
}

export default function Chart({ chartData, darkMode = false }: ChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 400;
    const height = 280;
    const margin = { top: 30, right: 30, bottom: 70, left: 50 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const textColor = darkMode ? "#e4e4e7" : "#27272a";
    const gridColor = darkMode ? "#3f3f46" : "#e4e4e7";

    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Title
    if (chartData.title) {
      svg
        .append("text")
        .attr("x", width / 2)
        .attr("y", 16)
        .attr("text-anchor", "middle")
        .attr("fill", textColor)
        .attr("font-size", "14px")
        .attr("font-weight", "600")
        .text(chartData.title);
    }

    switch (chartData.type) {
      case "line":
      case "scatter": {
        const points = chartData.data as { x: number; y: number }[];
        const xExtent = d3.extent(points, (d) => d.x) as [number, number];
        const yExtent = d3.extent(points, (d) => d.y) as [number, number];

        const x = d3.scaleLinear().domain(xExtent).nice().range([0, innerWidth]);
        const y = d3.scaleLinear().domain(yExtent).nice().range([innerHeight, 0]);

        // Grid
        g.append("g")
          .attr("class", "grid")
          .attr("transform", `translate(0,${innerHeight})`)
          .call(d3.axisBottom(x).tickSize(-innerHeight).tickFormat(() => ""))
          .selectAll("line")
          .attr("stroke", gridColor)
          .attr("stroke-opacity", 0.5);

        g.append("g")
          .attr("class", "grid")
          .call(d3.axisLeft(y).tickSize(-innerWidth).tickFormat(() => ""))
          .selectAll("line")
          .attr("stroke", gridColor)
          .attr("stroke-opacity", 0.5);

        // Axes
        g.append("g")
          .attr("transform", `translate(0,${innerHeight})`)
          .call(d3.axisBottom(x))
          .selectAll("text")
          .attr("fill", textColor);

        g.append("g")
          .call(d3.axisLeft(y))
          .selectAll("text")
          .attr("fill", textColor);

        g.selectAll(".domain").attr("stroke", textColor);
        g.selectAll(".tick line").attr("stroke", textColor);

        if (chartData.type === "line") {
          const line = d3
            .line<{ x: number; y: number }>()
            .x((d) => x(d.x))
            .y((d) => y(d.y));

          g.append("path")
            .datum(points)
            .attr("fill", "none")
            .attr("stroke", "#3b82f6")
            .attr("stroke-width", 2)
            .attr("d", line);
        }

        // Points
        g.selectAll("circle")
          .data(points)
          .join("circle")
          .attr("cx", (d) => x(d.x))
          .attr("cy", (d) => y(d.y))
          .attr("r", chartData.type === "scatter" ? 5 : 3)
          .attr("fill", "#3b82f6");

        break;
      }

      case "histogram": {
        const values = chartData.data as number[];
        const xExtent = d3.extent(values) as [number, number];

        const x = d3.scaleLinear().domain(xExtent).nice().range([0, innerWidth]);

        const histogram = d3
          .bin()
          .domain(x.domain() as [number, number])
          .thresholds(x.ticks(20));

        const bins = histogram(values);

        const y = d3
          .scaleLinear()
          .domain([0, d3.max(bins, (d) => d.length) || 0])
          .nice()
          .range([innerHeight, 0]);

        // Axes
        g.append("g")
          .attr("transform", `translate(0,${innerHeight})`)
          .call(d3.axisBottom(x))
          .selectAll("text")
          .attr("fill", textColor);

        g.append("g")
          .call(d3.axisLeft(y))
          .selectAll("text")
          .attr("fill", textColor);

        g.selectAll(".domain").attr("stroke", textColor);
        g.selectAll(".tick line").attr("stroke", textColor);

        // Bars
        g.selectAll("rect")
          .data(bins)
          .join("rect")
          .attr("x", (d) => x(d.x0 || 0) + 1)
          .attr("width", (d) => Math.max(0, x(d.x1 || 0) - x(d.x0 || 0) - 2))
          .attr("y", (d) => y(d.length))
          .attr("height", (d) => innerHeight - y(d.length))
          .attr("fill", "#3b82f6");

        break;
      }

      case "bar": {
        const items = chartData.data as { label: string; value: number }[];

        const x = d3
          .scaleBand()
          .domain(items.map((d) => d.label))
          .range([0, innerWidth])
          .padding(0.2);

        const y = d3
          .scaleLinear()
          .domain([0, d3.max(items, (d) => d.value) || 0])
          .nice()
          .range([innerHeight, 0]);

        // Axes
        g.append("g")
          .attr("transform", `translate(0,${innerHeight})`)
          .call(d3.axisBottom(x))
          .selectAll("text")
          .attr("fill", textColor)
          .attr("transform", "rotate(-45)")
          .attr("text-anchor", "end");

        g.append("g")
          .call(d3.axisLeft(y))
          .selectAll("text")
          .attr("fill", textColor);

        g.selectAll(".domain").attr("stroke", textColor);
        g.selectAll(".tick line").attr("stroke", textColor);

        // Bars
        g.selectAll("rect")
          .data(items)
          .join("rect")
          .attr("x", (d) => x(d.label) || 0)
          .attr("width", x.bandwidth())
          .attr("y", (d) => y(d.value))
          .attr("height", (d) => innerHeight - y(d.value))
          .attr("fill", "#3b82f6");

        break;
      }
    }

    // Axis labels
    if (chartData.xLabel) {
      svg
        .append("text")
        .attr("x", width / 2)
        .attr("y", height - 5)
        .attr("text-anchor", "middle")
        .attr("fill", textColor)
        .attr("font-size", "12px")
        .text(chartData.xLabel);
    }

    if (chartData.yLabel) {
      svg
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -height / 2)
        .attr("y", 12)
        .attr("text-anchor", "middle")
        .attr("fill", textColor)
        .attr("font-size", "12px")
        .text(chartData.yLabel);
    }
  }, [chartData, darkMode]);

  return (
    <svg
      ref={svgRef}
      className="w-full max-w-lg pb-2"
      style={{ minHeight: "280px" }}
    />
  );
}
