# core/analytics/reports/report_exporter.py

from pathlib import Path
from datetime import datetime
import json
import pandas as pd

# PDF (REQUIRED LIBRARY)
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-index columns for export."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(map(str, c)).strip() for c in df.columns]
    return df.reset_index(drop=False)


def _df_to_table(df: pd.DataFrame):
    """Convert dataframe into ReportLab table."""
    data = [df.columns.tolist()] + df.astype(str).values.tolist()

    table = Table(data, repeatRows=1)

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    return table


# ---------------------------------------------------------
# EXPORT ENGINE
# ---------------------------------------------------------

class LearningReportExporter:
    """
    Converts LearningReport into publication-ready artifacts.

    Outputs:
        /exports/
            report_TIMESTAMP.pdf
            report_TIMESTAMP.json
            cohort_summary.csv
            competency_summary.csv
            modality_alignment.csv
    """

    def __init__(self, report, export_dir="exports"):

        self.report = report
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # -----------------------------------------------------
    # CSV EXPORT
    # -----------------------------------------------------
    def export_csv(self):

        outputs = {}

        if not self.report.cohort_summary.empty:
            path = self.export_dir / f"cohort_summary_{self.timestamp}.csv"
            _flatten_columns(self.report.cohort_summary).to_csv(
                path, index=False
            )
            outputs["cohort_summary"] = str(path)

        if not self.report.competency_summary.empty:
            path = self.export_dir / f"competency_summary_{self.timestamp}.csv"
            self.report.competency_summary.to_csv(path, index=False)
            outputs["competency_summary"] = str(path)

        if not self.report.modality_alignment.empty:
            path = self.export_dir / f"modality_alignment_{self.timestamp}.csv"
            self.report.modality_alignment.to_csv(path, index=False)
            outputs["modality_alignment"] = str(path)

        return outputs

    # -----------------------------------------------------
    # JSON EXPORT (RESEARCH REPRODUCIBILITY)
    # -----------------------------------------------------
    def export_json(self):

        payload = {
            "diagnostics": self.report.diagnostics,
            "cohort_summary": _flatten_columns(
                self.report.cohort_summary
            ).to_dict(orient="records")
            if not self.report.cohort_summary.empty
            else [],
            "competency_summary": self.report.competency_summary.to_dict(
                orient="records"
            )
            if not self.report.competency_summary.empty
            else [],
            "modality_alignment": self.report.modality_alignment.to_dict(
                orient="records"
            )
            if not self.report.modality_alignment.empty
            else [],
        }

        path = self.export_dir / f"learning_report_{self.timestamp}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return str(path)

    # -----------------------------------------------------
    # PDF EXPORT (PUBLICATION READY)
    # -----------------------------------------------------
    def export_pdf(self):

        path = self.export_dir / f"learning_report_{self.timestamp}.pdf"

        doc = SimpleDocTemplate(
            str(path),
            pagesize=LETTER,
            title="Learning Analytics Report",
        )

        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(
            Paragraph(
                "Learning Analytics Report",
                styles["Title"],
            )
        )

        story.append(
            Paragraph(
                f"Generated: {datetime.now().isoformat()}",
                styles["Normal"],
            )
        )

        story.append(Spacer(1, 12))

        # Diagnostics
        story.append(Paragraph("Diagnostics", styles["Heading2"]))

        for k, v in self.report.diagnostics.items():
            story.append(
                Paragraph(f"<b>{k}</b>: {v}", styles["Normal"])
            )

        story.append(Spacer(1, 12))

        # Cohort Summary
        if not self.report.cohort_summary.empty:
            story.append(
                Paragraph("Cohort Summary", styles["Heading2"])
            )
            df = _flatten_columns(self.report.cohort_summary)
            story.append(_df_to_table(df))
            story.append(Spacer(1, 12))

        # Competency Ranking
        if not self.report.competency_summary.empty:
            story.append(
                Paragraph("Competency Ranking", styles["Heading2"])
            )
            story.append(
                _df_to_table(self.report.competency_summary)
            )
            story.append(Spacer(1, 12))

        # Alignment
        if not self.report.modality_alignment.empty:
            story.append(
                Paragraph(
                    "Quantitative–Qualitative Alignment",
                    styles["Heading2"],
                )
            )
            story.append(
                _df_to_table(self.report.modality_alignment)
            )

        doc.build(story)

        return str(path)

    # -----------------------------------------------------
    # MASTER EXPORT
    # -----------------------------------------------------
    def export_all(self):

        results = {}

        results["csv"] = self.export_csv()
        results["json"] = self.export_json()
        results["pdf"] = self.export_pdf()

        return results