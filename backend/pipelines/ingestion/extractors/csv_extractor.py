import pandas as pd
from typing import List

from monitoring.tracing import trace_sync

@trace_sync("ingestion.extract.csv")
def extract_csv(file_path: str) -> List[ExtractedPage]:
    df = pd.read_csv(file_path)
    pages = []
    rows = len(df)
    
    if rows < 1000:
        # Convert entire thing to markdown table
        content = df.to_markdown(index=False)
        pages.append(ExtractedPage(
            page_number=1,
            content=content,
            content_type="table",
            metadata={"rows": rows}
        ))
    else:
        # Generate summary per 100 rows
        for i in range(0, rows, 100):
            chunk = df.iloc[i:i+100]
            
            # Basic stats
            summary_parts = [
                f"Data chunk rows {i} to {i+min(100, len(chunk)-1)}",
                f"Column names: {', '.join(chunk.columns)}",
                f"Dtypes:\n{chunk.dtypes.to_string()}"
            ]
            
            # Numeric stats
            numeric_cols = chunk.select_dtypes(include=['number'])
            if not numeric_cols.empty:
                summary_parts.append(f"Numeric Min/Max/Mean:\n{numeric_cols.agg(['min', 'max', 'mean']).to_string()}")
                
            # Categorical stats
            cat_cols = chunk.select_dtypes(include=['object', 'category'])
            if not cat_cols.empty:
                cat_summary = []
                for col in cat_cols.columns:
                    top_5 = chunk[col].value_counts().head(5).to_dict()
                    cat_summary.append(f"{col} top 5: {top_5}")
                summary_parts.append("Categorical Top 5:\n" + "\n".join(cat_summary))
                
            # Sample rows
            summary_parts.append(f"Sample Rows:\n{chunk.head(3).to_markdown(index=False)}")
            
            pages.append(ExtractedPage(
                page_number=(i // 100) + 1,
                content="\n\n".join(summary_parts),
                content_type="text",
                metadata={"chunk_start": i, "chunk_size": len(chunk)}
            ))
            
    return pages
