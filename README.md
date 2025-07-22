# GSC Performance & SERP Overlap Analyzer

A comprehensive Python application built for Streamlit that analyzes Google Search Console performance data to identify query overlaps between URLs and conduct SERP overlap analysis using the Serper API. This tool helps detect potential keyword cannibalization issues through sophisticated overlap calculations.

## 🚀 Features

### Core Analysis Capabilities
- **Query Overlap Analysis**: Groups unique URLs and calculates shared query percentages between URL pairs
- **SERP Overlap Analysis**: Scrapes top 10 organic results using Serper API and calculates SERP similarity
- **Dual Reporting System**: Generates both URL-based and Query-based reports
- **Cannibalization Detection**: Identifies potential keyword cannibalization issues with configurable thresholds

### Technical Features
- ✅ **Python 3.12 Compatible**: Built specifically for Python 3.12 and Streamlit
- ✅ **Modular Architecture**: Clean, maintainable codebase with separate utility modules
- ✅ **Smart Data Validation**: Handles different GSC export formats automatically
- ✅ **Rate Limiting**: Intelligent API throttling to prevent service disruptions
- ✅ **Progress Tracking**: Real-time progress indicators for long-running analyses
- ✅ **Error Recovery**: Robust error handling with detailed user feedback
- ✅ **Performance Optimization**: Caching and efficient data processing
- ✅ **Cost Management**: Configurable limits to control API usage costs

## 📋 Requirements

- Python 3.12+
- Streamlit 1.28.0+
- Valid Serper API key (get from [serper.dev](https://serper.dev))
- Google Search Console performance data export (CSV format)

## 🛠 Installation

### 1. Clone or Create Project Directory
```bash
mkdir gsc-serp-analyzer
cd gsc-serp-analyzer
```

### 2. Create Virtual Environment
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Create `requirements.txt`:
```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
plotly>=5.15.0
```

Install packages:
```bash
pip install -r requirements.txt
```

### 4. Create Project Structure
```
gsc-serp-analyzer/
├── streamlit_app.py          # Main application
├── requirements.txt          # Dependencies
├── README.md                # This file
└── utils/
    ├── __init__.py
    ├── gsc_processor.py      # GSC data handling
    ├── serp_scraper.py       # Serper API integration
    ├── overlap_calculator.py # Overlap algorithms
    └── report_generator.py   # Report creation
```

## ⚙️ Setup & Configuration

### 1. Get Serper API Key
1. Visit [serper.dev](https://serper.dev)
2. Create an account and obtain your API key
3. Note: Serper API costs approximately $1 per 1000 searches

### 2. Export GSC Performance Data
Export your Google Search Console performance data with these required columns:
- **Query/Queries**: Search terms
- **URL/Page**: Landing page URLs
- **Clicks**: Number of clicks
- **Impressions**: Number of impressions
- **CTR**: Click-through rate
- **Position**: Average search position

**Export Instructions:**
1. Go to Google Search Console
2. Navigate to Performance > Search Results
3. Set your desired date range
4. Click "Export" and choose CSV format
5. Ensure all required dimensions and metrics are included

## 🚀 Usage

### 1. Launch Application
```bash
streamlit run streamlit_app.py
```

### 2. Configure Analysis
1. **Upload GSC Data**: Use the sidebar file uploader for your CSV export
2. **Enter API Key**: Input your Serper API key in the configuration section
3. **Set Parameters**:
   - **Minimum Clicks**: Filter queries (default: 1 click minimum)
   - **SERP Overlap Threshold**: Set threshold for high overlap (default: 50%)
   - **Max Queries**: Limit analysis to control API costs (default: 100)

### 3. Run Analysis
1. Click "Start Analysis" after configuration
2. Monitor progress through real-time indicators
3. Review results in organized tabs
4. Download reports as CSV files

## 📊 Analysis Types

### 1. Query Overlap Analysis
**Purpose**: Identifies shared queries between URL pairs

**How it works**:
- Groups all unique URLs from your GSC data
- Cross-matches every query each URL ranks for
- Calculates overlap numbers and percentages

**Example**:
- URL A ranks for 100 queries
- URL B ranks for 200 queries
- URL A shares 80 queries with URL B = **80% overlap for URL A**
- URL B shares 80 queries with URL A = **40% overlap for URL B**

### 2. SERP Overlap Analysis
**Purpose**: Measures competition level between queries based on SERP similarity

**How it works**:
- Scrapes top 10 organic results for each query (queries with ≥1 click only)
- Compares SERP results between query pairs
- Calculates similarity percentages using Jaccard similarity
- Identifies queries competing for similar SERP real estate

## 📈 Report Outputs

### URL-Based Report
**Columns**:
- URL
- Total Clicks
- Total Impressions
- Unique Queries Count
- Top Queries (sample)
- High SERP Overlaps Count (>50% threshold)
- Query Overlap Statistics

**Use Case**: High-level cannibalization overview at URL level

### Query-Based Report
**Exact Structure as Requested**:
- **Column 1**: URLs
- **Column 2**: Queries
- **Column 3**: SERP Overlap %

**Additional Context**:
- Clicks, Impressions, CTR, Position data
- Competing URLs information
- Overlap severity indicators

**Use Case**: Detailed, granular analysis for specific optimization decisions

## 🔧 Configuration Options

### Analysis Parameters
```python
# Configurable in the UI
MIN_CLICKS = 1              # Minimum clicks for SERP analysis
SERP_OVERLAP_THRESHOLD = 50 # Percentage threshold for high overlap
MAX_QUERIES = 100           # Maximum queries to analyze (cost control)
API_DELAY = 1               # Seconds between API calls
```

### Data Filtering Rules
1. **Query Overlap Calculation**: Includes ALL queries (including 0-click)
2. **SERP Overlap Calculation**: Only queries with ≥1 click
3. **Report Generation**: Configurable thresholds and limits

## 🔍 Understanding the Results

### High Query Overlap (URL-Based)
- **80%+ Overlap**: Strong cannibalization indicator
- **50-79% Overlap**: Moderate cannibalization risk
- **<50% Overlap**: Low risk, but monitor trends

### High SERP Overlap (Query-Based)
- **70%+ SERP Overlap**: Very high competition between queries
- **50-69% SERP Overlap**: Moderate competition
- **<50% SERP Overlap**: Different search intents

### Prioritization Strategy
1. **Start with URL-Based Report**: Identify problematic URL pairs
2. **Drill down with Query-Based Report**: Understand specific query conflicts
3. **Focus on High Clicks + High Overlap**: Maximum impact optimization opportunities

## 💰 API Cost Management

### Cost Calculation
- Serper API: ~$1 per 1000 searches
- Analysis of 100 queries ≈ $0.10
- Analysis of 1000 queries ≈ $1.00

### Cost Control Features
- **Query Limits**: Set maximum queries to analyze
- **Click Filtering**: Only analyze queries with meaningful traffic
- **Progress Monitoring**: Real-time cost tracking
- **Batch Processing**: Efficient API usage

### Optimization Tips
1. Start with higher click thresholds for initial analysis
2. Use query limits during exploratory analysis
3. Run full analysis only on prioritized query sets
4. Monitor API usage in Serper dashboard

## 🐛 Troubleshooting

### Common Issues

**"API Key Invalid"**
- Verify your Serper API key is correct
- Check API key permissions and quota
- Ensure no extra spaces or characters

**"CSV Upload Failed"**
- Verify CSV has required columns (Query, URL, Clicks, etc.)
- Check for special characters in column headers
- Ensure file is not corrupted

**"Analysis Timeout"**
- Reduce the number of queries to analyze
- Increase minimum click threshold
- Check internet connection stability

**"High Memory Usage"**
- Process smaller data batches
- Increase minimum click threshold
- Clear browser cache and restart application

### Performance Tips
1. **Large Datasets**: Process in smaller batches
2. **Memory Issues**: Increase click thresholds
3. **API Limits**: Use built-in rate limiting
4. **Browser Issues**: Use Chrome/Firefox for best performance

## 📝 Data Requirements & Formats

### Required GSC Columns
| Column Name | Alternative Names | Description |
|-------------|------------------|-------------|
| Query | Queries, Search Term | Search queries/keywords |
| URL | Page, Landing Page | Landing page URLs |
| Clicks | Click Count | Number of clicks |
| Impressions | Impression Count | Number of impressions |
| CTR | Click Through Rate | Click-through rate (%) |
| Position | Avg Position | Average search position |

### Data Quality Tips
- Export data for meaningful date ranges (30-90 days recommended)
- Include all devices and search types for comprehensive analysis
- Ensure no data sampling is applied in GSC
- Remove any manually filtered dimensions before export

## 🚦 Best Practices

### Analysis Approach
1. **Start Small**: Begin with top-performing queries
2. **Iterate**: Refine thresholds based on initial results
3. **Validate**: Cross-reference findings with manual SERP checks
4. **Document**: Keep records of analysis parameters and findings

### Optimization Workflow
1. **Identify**: Use reports to find cannibalization issues
2. **Prioritize**: Focus on high-traffic, high-overlap scenarios
3. **Implement**: Make content or technical changes
4. **Monitor**: Re-run analysis to measure improvements

## 🔄 Updates & Maintenance

### Regular Tasks
- **Monthly Analysis**: Run comprehensive overlap analysis
- **API Key Management**: Monitor usage and quotas
- **Data Refresh**: Update with latest GSC exports
- **Threshold Tuning**: Adjust based on site performance

### Version Updates
- Check for Streamlit updates compatibility
- Monitor Serper API changes and pricing
- Update dependencies regularly for security

## 📞 Support & Documentation

### Additional Resources
- [Google Search Console Help](https://support.google.com/webmasters/)
- [Serper API Documentation](https://serper.dev/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Contributing
This is a modular application designed for easy extension. Key areas for enhancement:
- Additional SERP APIs integration
- Advanced overlap algorithms
- Enhanced visualization features
- Automated reporting schedules

---

**Created for Python 3.12 + Streamlit Environment**  
*Designed for SEO professionals and digital marketers to identify and resolve keyword cannibalization issues through data-driven analysis.*