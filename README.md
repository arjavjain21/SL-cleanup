# Smart Lead Cleanup 🧹

A comprehensive, safety-first solution for cleaning up inactive leads from Smartlead campaigns with built-in backup, validation, and reporting features.

## 🚀 Features

### ✅ **Safety First**
- **Automatic Backup**: Creates compressed backups before any deletion
- **Pre-deletion Validation**: Re-validates leads before deletion to prevent race conditions
- **Dry-run Mode**: Test without actually deleting anything
- **Confirmation Prompts**: Requires explicit confirmation for destructive operations
- **Rollback Support**: Audit trails for potential data recovery

### ⚡ **Performance**
- **Async/Concurrent Processing**: Up to 10x faster with configurable concurrency
- **Intelligent Batching**: Processes deletions in manageable batches
- **Smart Campaign Selection**: Dynamically selects campaigns to reach target
- **Progress Tracking**: Real-time progress updates with ETA

### 📊 **Monitoring & Reporting**
- **Comprehensive Email Reports**: Detailed execution summaries
- **Audit Trails**: JSON logs of all operations
- **Error Handling**: Robust retry logic and error categorization
- **Statistics Tracking**: Success rates, performance metrics

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- Valid Smartlead API key
- Gmail account with App Password (for email reports)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/SL-cleanup.git
cd SL-cleanup

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your credentials
```

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SMARTLEAD_API_KEY` | ✅ | - | Your Smartlead API key |
| `TARGET_LEADS` | ❌ | 30000 | Target number of leads to delete |
| `DAYS_THRESHOLD` | ❌ | 45 | Minimum days since campaign creation |
| `MAX_CONCURRENCY` | ❌ | 10 | Maximum concurrent API requests |
| `EXCLUDE_CLIENT_IDS` | ❌ | 1598 | Comma-separated client IDs to exclude |
| `SENDER_EMAIL` | ❌ | - | Gmail address for sending reports |
| `SMTP_PASSWORD` | ❌ | - | Gmail App Password |
| `RECIPIENT_EMAILS` | ❌ | - | Comma-separated recipient emails |

### Gmail App Password Setup
1. Enable 2-factor authentication on your Gmail account
2. Go to Google Account settings > Security > 2-Step Verification > App passwords
3. Generate an app password for "Mail"
4. Use this password in `SMTP_PASSWORD`

## 🎮 Usage

### Basic Usage
```bash
# Dry run (recommended first)
python smart_lead_cleanup.py --dry-run

# Production run
python smart_lead_cleanup.py

# Custom parameters
python smart_lead_cleanup.py --target-leads 50000 --days-threshold 30
```

### Command Line Options
- `--dry-run`: Simulate without actual deletion
- `--target-leads N`: Override target number of leads
- `--days-threshold N`: Override days threshold
- `--max-concurrency N`: Override concurrency limit

## 🔒 Safety Mechanisms

### 1. **Backup System**
- Creates compressed `.csv.gz` backups before any deletion
- Includes metadata (timestamp, reason, configuration)
- Stored in `backups/` directory

### 2. **Pre-deletion Validation**
- Re-checks each lead's reply count before deletion
- Skips leads that gained replies since export
- Prevents deletion of engaged leads

### 3. **Confirmation System**
```
⚠️  DANGER: About to delete 25,847 leads permanently!
This action CANNOT be undone.
Backup will be created in: backups/

Type 'DELETE_CONFIRMED' to proceed:
```

### 4. **Audit Trail**
- JSON logs of every operation
- Timestamps, success/failure status
- Error details and retry attempts

## 📈 Campaign Selection Logic

The script intelligently selects campaigns to process:

1. **Filters campaigns** by:
   - Status: `PAUSED` or `COMPLETED`
   - Age: Created more than `DAYS_THRESHOLD` days ago
   - Excludes specific client IDs

2. **Prioritizes campaigns** with highest no-reply lead counts

3. **Dynamically selects** campaigns until reaching `TARGET_LEADS`

## 📧 Email Reports

Comprehensive reports include:
- Execution summary and statistics
- Processing metrics and success rates
- Backup file information
- Attached files: backup, audit log, execution log

## 📁 Directory Structure

```
SL-cleanup/
├── smart_lead_cleanup.py     # Main script
├── requirements.txt          # Dependencies
├── .env.example             # Environment template
├── README.md                # Documentation
├── backups/                 # Backup files (auto-created)
├── logs/                    # Log files (auto-created)
└── exports/                 # Export files (auto-created)
```

## 🔧 GitHub Actions Integration

The script is designed to work seamlessly with GitHub Actions:

### Secrets to Configure
- `SMARTLEAD_API_KEY`
- `SENDER_EMAIL`
- `SMTP_PASSWORD`
- `RECIPIENT_EMAILS`

### Example Workflow (create `.github/workflows/lead-cleanup.yml`)
```yaml
name: Smart Lead Cleanup

on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday at 2 AM
  workflow_dispatch:     # Manual trigger

jobs:
  cleanup:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: pip install -r requirements.txt
    
    - name: Run lead cleanup
      env:
        SMARTLEAD_API_KEY: ${{ secrets.SMARTLEAD_API_KEY }}
        SENDER_EMAIL: ${{ secrets.SENDER_EMAIL }}
        SMTP_PASSWORD: ${{ secrets.SMTP_PASSWORD }}
        RECIPIENT_EMAILS: ${{ secrets.RECIPIENT_EMAILS }}
      run: python smart_lead_cleanup.py
    
    - name: Upload logs
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: cleanup-logs
        path: |
          logs/
          backups/
```

## 🚨 Error Handling

The script handles various scenarios:
- **API failures**: Exponential backoff retry
- **Network issues**: Connection pooling and timeouts
- **Rate limits**: Intelligent throttling
- **Partial failures**: Batch-level error isolation
- **Data inconsistencies**: Validation and logging

## 📊 Performance

Typical performance metrics:
- **Processing speed**: 500-1000 deletions per minute
- **Concurrency**: 10 simultaneous API requests
- **Memory usage**: ~100MB for 30k leads
- **Execution time**: 30-60 minutes for 30k leads

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool performs destructive operations on your Smartlead data. While comprehensive safety measures are in place, always:
1. Test with `--dry-run` first
2. Verify backup files are created
3. Keep backups in multiple locations
4. Monitor execution reports

Use at your own risk. The authors are not responsible for data loss.

## 🆘 Support

For issues, questions, or feature requests:
1. Check existing GitHub issues
2. Create a new issue with detailed description
3. Include log files and configuration (redacted)

---


**Made with ❤️ for safer lead management**
