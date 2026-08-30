"""Utilities for extracting and using metadata from questions and filenames."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================================
# Filename Metadata Extraction (for ingestion)
# ============================================================================

@dataclass
class DocumentMetadata:
    """Metadata extracted from a FinanceBench PDF filename."""
    company: str
    year: int
    quarter: Optional[str]  # Q1, Q2, Q3, Q4 or None for annual
    doc_type: str  # 10K, 10Q, 8K, EARNINGS, or ANNUALREPORT
    source_file: str

    def to_dict(self) -> dict:
        """Convert to dictionary for ChromaDB metadata."""
        return {
            "company": self.company,
            "year": self.year,
            "quarter": self.quarter or "",
            "doc_type": self.doc_type,
            "source_file": self.source_file,
            "fiscal_period": f"FY{self.year}" if not self.quarter else f"FY{self.year}{self.quarter}",
        }


def parse_filename(filename: str) -> Optional[DocumentMetadata]:
    """
    Parse a FinanceBench PDF filename to extract metadata.

    Handles FinanceBench 10-K/10-Q, 8-K, earnings, and annual-report names.
    Examples:
        - 3M_2018_10K.pdf -> {company: "3M", year: 2018, quarter: None, doc_type: "10K"}
        - ADOBE_2022Q2_10Q.pdf -> {company: "ADOBE", year: 2022, quarter: "Q2", doc_type: "10Q"}

    Args:
        filename: PDF filename (e.g., "3M_2018_10K.pdf")

    Returns:
        DocumentMetadata object or None if parsing fails
    """
    # Remove .pdf extension
    name = Path(filename).stem

    year_match = re.search(
        r"(?:^|_)(20\d{2})(?:Q([1-4]))?(?=_|$)",
        name,
        re.IGNORECASE,
    )
    doc_type_match = re.search(
        r"(?:^|_)(10K|10Q|8K|EARNINGS|ANNUALREPORT)(?=_|-|$)",
        name,
        re.IGNORECASE,
    )
    if year_match is None:
        year_match = re.search(r"dated[-_](20\d{2})", name, re.IGNORECASE)

    if year_match is None or doc_type_match is None:
        return None

    company_end = min(year_match.start(), doc_type_match.start())
    company = name[:company_end].strip("_")
    year = year_match.group(1)
    quarter_number = year_match.group(2) if year_match.lastindex == 2 else None
    quarter = f"Q{quarter_number}" if quarter_number else None
    doc_type = doc_type_match.group(1)

    return DocumentMetadata(
        company=normalize_company_name(company),
        year=int(year),
        quarter=quarter.upper() if quarter else None,
        doc_type=doc_type.upper(),
        source_file=filename,
    )


# Company name normalization mapping
COMPANY_MAPPINGS = {
    # Aliases whose normalized question name differs from the FinanceBench
    # filename key stored during ingestion.
    "AESCORPORATION": "AES",
    "ACTIVSIONBLIZZARD": "ACTIVISIONBLIZZARD",  # Dataset typo alias
    "ADVANCEDMICRODEVICES": "AMD",
    "AMEX": "AMERICANEXPRESS",
    "AMERICANWATER": "AMERICANWATERWORKS",
    "JANDJ": "JOHNSON_JOHNSON",
    "JNJ": "JOHNSON_JOHNSON",
    "JOHNSONANDJOHNSON": "JOHNSON_JOHNSON",
    "JOHNSONJOHNSON": "JOHNSON_JOHNSON",
    "JPMORGANCHASE": "JPMORGAN",
    "JPM": "JPMORGAN",
    "MGM": "MGMRESORTS",
    "PGE": "PG_E",
    "PGANDE": "PG_E",
    "PGANDECORPORATION": "PG_E",
    "PGECORPORATION": "PG_E",
    "SQUARE": "BLOCK",
}


def normalize_company_name(company: str) -> str:
    """Return the canonical company key used in ingested metadata."""

    normalized = company.upper().replace("&", "AND")
    normalized = re.sub(r"[^A-Z0-9]", "", normalized)
    return COMPANY_MAPPINGS.get(normalized, normalized)


# ============================================================================
# Question Metadata Extraction (for retrieval)
# ============================================================================

def extract_metadata_from_question(question: str) -> Dict[str, any]:
    """Extract company name, year, and document type from question.

    Args:
        question: Question text

    Returns:
        Dict with extracted metadata: company, years, doc_type
    """
    metadata = {
        'companies': [],
        'years': [],
        'doc_types': []
    }

    # All companies in FinanceBench dataset (including variations)
    companies = [
        # Original list
        '3M', 'Adobe', 'Apple', 'Microsoft', 'Amazon', 'Netflix',
        'Oracle', 'Block', 'Square', 'Costco', 'CVS', 'AES',
        'Activision Blizzard', 'American Express', 'AMEX', 'Best Buy',
        'Coca-Cola', 'Coca Cola', 'Boeing', 'Pfizer', 'Walmart',
        # Additional companies from FinanceBench
        'AMD', 'Advanced Micro Devices',
        'Amcor', 'AMCOR',
        'American Water Works', 'American Water',
        'Corning',
        'CVS Health', 'CVSHEALTH',
        'General Mills', 'GeneralMills',
        'JPMorgan', 'JP Morgan', 'JPMorgan Chase', 'JPM',
        'MGM Resorts', 'MGM', 'MGMRESORTS',
        'Nike', 'NIKE',
        'PayPal', 'Paypal',
        'Ulta Beauty', 'Ulta', 'ULTABEAUTY',
        'Verizon',
        # Companies from missing PDFs (for future)
        'Johnson & Johnson', 'Johnson and Johnson', 'J&J', 'JnJ',
        'PepsiCo', 'Pepsi',
        'Lockheed Martin', 'Lockheed',
        'Kraft Heinz', 'Kraft',
        'Foot Locker', 'FootLocker',
    ]

    # Extract companies (case insensitive)
    question_lower = question.lower()
    for company in companies:
        if company.lower() in question_lower:
            metadata['companies'].append(company)

    # Extract years - use comprehensive pattern that captures all formats
    # and prioritizes 4-digit years to avoid FY2022 being parsed as FY20 (year 2020)
    years_found = set()

    # First pass: Match FY + 4-digit year (FY2019, FY 2019)
    fy_4digit = re.findall(r'FY\s?(\d{4})', question, re.IGNORECASE)
    for year_str in fy_4digit:
        years_found.add(int(year_str))

    # Second pass: Match standalone 4-digit years (2019, 2020)
    standalone_4digit = re.findall(r'\b(20\d{2})\b', question, re.IGNORECASE)
    for year_str in standalone_4digit:
        years_found.add(int(year_str))

    # Third pass: Match FY + 2-digit year ONLY if no 4-digit FY year was found
    if not fy_4digit:
        fy_2digit = re.findall(r'FY\s?(\d{2})', question, re.IGNORECASE)
        for year_str in fy_2digit:
            year = int(year_str)
            full_year = 2000 + year
            years_found.add(full_year)

    # Fourth pass: Match abbreviated years ('19, '20)
    abbreviated = re.findall(r"'(\d{2})\b", question)
    for year_str in abbreviated:
        year = int(year_str)
        full_year = 2000 + year
        years_found.add(full_year)

    metadata['years'] = sorted(list(years_found))

    # Extract document types
    if any(term in question_lower for term in ['10-k', '10k', 'annual report']):
        metadata['doc_types'].append('10k')
    if any(term in question_lower for term in ['10-q', '10q', 'quarterly']):
        metadata['doc_types'].append('10q')
    if any(term in question_lower for term in ['8-k', '8k']):
        metadata['doc_types'].append('8k')

    return metadata


def filter_chunks_by_metadata(chunks: List, metadata: Dict) -> List:
    """Filter retrieved chunks based on extracted metadata.

    Uses chunk metadata fields (company, year) for STRICT matching,
    not just source filename strings.

    Args:
        chunks: List of Document objects from retriever
        metadata: Extracted metadata from question (companies, years, doc_types)

    Returns:
        Filtered list of chunks matching the metadata criteria
    """
    if not chunks or not metadata:
        return chunks

    # Extract filter criteria (with safe .get())
    target_companies = {
        normalize_company_name(c) for c in metadata.get('companies', [])
    }
    target_years = metadata.get('years', [])
    target_doc_types = [d.upper() for d in metadata.get('doc_types', [])]

    filtered = []

    for chunk in chunks:
        chunk_meta = chunk.metadata
        source = chunk_meta.get('source', '').lower()
        source_metadata = parse_filename(Path(source).name) if source else None

        # Get chunk's actual metadata (prefer metadata fields over filename parsing)
        chunk_company = normalize_company_name(chunk_meta.get('company', ''))
        chunk_year = chunk_meta.get('year')

        # STRICT company match - use metadata field if available, else filename
        company_match = True
        if target_companies:
            if chunk_company:
                # Use actual metadata field
                company_match = chunk_company in target_companies
            elif source_metadata is not None:
                company_match = source_metadata.company in target_companies
            else:
                # Fallback to filename matching
                canonical_source = normalize_company_name(source)
                company_match = any(c in canonical_source for c in target_companies)

        # STRICT year match - use metadata field if available, else filename
        year_match = True
        if target_years:
            if chunk_year is not None:
                # Use actual metadata field
                year_match = chunk_year in target_years
            elif source_metadata is not None:
                year_match = source_metadata.year in target_years
            else:
                # Fallback to filename matching
                year_match = any(str(y) in source for y in target_years)

        # Doc type match (filename only - not stored in metadata)
        doctype_match = True
        if target_doc_types:
            doctype_match = any(d.lower() in source for d in target_doc_types)

        # Include chunk if it matches all filters
        if company_match and year_match and doctype_match:
            filtered.append(chunk)

    # High-confidence metadata must fail closed. Returning unrelated top-ranked
    # chunks can produce a fluent answer about the wrong company or period.
    if not filtered:
        print(f"⚠️ Metadata filter: No chunks matched company={target_companies} year={target_years}")
        return []

    return filtered
