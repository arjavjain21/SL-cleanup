#!/usr/bin/env python3
"""
Smart Lead Cleanup - Consolidated Safety Script
Safely identifies and deletes inactive leads with comprehensive backup and validation
"""

import asyncio
import aiohttp
import argparse
import csv
import json
import logging
import os
import smtplib
import ssl
import sys
import time
import gzip
import shutil
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import pytz
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Configuration
@dataclass
class Config:
    # API Configuration
    api_key: str
    base_url: str = "https://server.smartlead.ai/api/v1"
    
    # Processing Configuration
    target_leads: int = 30000
    days_threshold: int = 45
    max_concurrency: int = 10
    exclude_client_ids: set = None
    
    # Email Configuration
    sender_email: str = ""
    smtp_password: str = ""
    recipient_emails: List[str] = None
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 465
    
    # Runtime Configuration
    dry_run: bool = False
    backup_dir: str = "backups"
    logs_dir: str = "logs"
    exports_dir: str = "exports"
    
    def __post_init__(self):
        if self.exclude_client_ids is None:
            self.exclude_client_ids = {1598}
        if self.recipient_emails is None:
            self.recipient_emails = []

class SmartLeadCleanup:
    def __init__(self, config: Config):
        self.config = config
        self.session = None
        self.logger = None
        self.audit_log = []
        self.stats = {
            'campaigns_processed': 0,
            'leads_exported': 0,
            'leads_backed_up': 0,
            'leads_deleted_success': 0,
            'leads_deleted_failed': 0,
            'api_errors': 0,
            'validation_failures': 0
        }
        self.start_time = None
        
        # Setup directories
        Path(self.config.backup_dir).mkdir(exist_ok=True)
        Path(self.config.logs_dir).mkdir(exist_ok=True)
        Path(self.config.exports_dir).mkdir(exist_ok=True)
        
        self._setup_logging()

    def _setup_logging(self):
        """Setup comprehensive logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{self.config.logs_dir}/smart_lead_cleanup_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file

    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=self.config.max_concurrency)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def _log_audit(self, action: str, details: dict):
        """Add entry to audit trail"""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        }
        self.audit_log.append(audit_entry)

    async def _make_request(self, method: str, url: str, params: dict = None, 
                          data: dict = None, max_retries: int = 3) -> Optional[aiohttp.ClientResponse]:
        """Make HTTP request with retry logic and error handling"""
        params = params or {}
        params['api_key'] = self.config.api_key
        
        for attempt in range(1, max_retries + 1):
            try:
                async with self.session.request(method, url, params=params, json=data) as response:
                    if response.status == 200:
                        return response
                    elif response.status == 404 and method == "DELETE":
                        self.logger.warning(f"Resource not found (404) - may already be deleted: {url}")
                        return response
                    else:
                        self.logger.error(f"HTTP {response.status} on attempt {attempt}: {url}")
                        self.stats['api_errors'] += 1
                        
            except asyncio.TimeoutError:
                self.logger.error(f"Timeout on attempt {attempt}: {url}")
            except Exception as e:
                self.logger.error(f"Request error on attempt {attempt}: {e}")
                self.stats['api_errors'] += 1
            
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return None

    async def fetch_campaigns(self) -> List[dict]:
        """Fetch all campaigns from API"""
        self.logger.info("Fetching campaigns from API...")
        
        url = f"{self.config.base_url}/campaigns"
        response = await self._make_request("GET", url)
        
        if response:
            campaigns = await response.json()
            self.logger.info(f"Retrieved {len(campaigns)} campaigns")
            self._log_audit("FETCH_CAMPAIGNS", {"count": len(campaigns)})
            return campaigns
        
        self.logger.error("Failed to fetch campaigns")
        return []

    def filter_campaigns(self, campaigns: List[dict]) -> List[dict]:
        """Filter campaigns based on criteria"""
        self.logger.info(f"Filtering campaigns (days_threshold={self.config.days_threshold}, exclude_clients={self.config.exclude_client_ids})")
        
        cutoff_date = datetime.now(pytz.timezone("Asia/Kolkata")) - timedelta(days=self.config.days_threshold)
        filtered = []
        
        # Create campaign summary CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_csv = f"{self.config.exports_dir}/campaign_summary_{timestamp}.csv"
        
        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Campaign ID", "Name", "Status", "Client ID",
                "Created At (UTC)", "Created At (IST)",
                "Updated At (UTC)", "Updated At (IST)",
                "Included in Filtered List"
            ])
            
            for campaign in campaigns:
                try:
                    client_id = campaign.get("client_id")
                    status = campaign.get("status")
                    created_utc = datetime.strptime(campaign["created_at"], "%Y-%m-%dT%H:%M:%S.%f%z")
                    updated_utc = datetime.strptime(campaign["updated_at"], "%Y-%m-%dT%H:%M:%S.%f%z")
                    created_ist = created_utc.astimezone(pytz.timezone("Asia/Kolkata"))
                    updated_ist = updated_utc.astimezone(pytz.timezone("Asia/Kolkata"))
                    
                    # Apply filters
                    included = False
                    if client_id not in self.config.exclude_client_ids:
                        if status in ("PAUSED", "COMPLETED") and created_ist <= cutoff_date:
                            included = True
                            filtered.append(campaign)
                    
                    writer.writerow([
                        campaign["id"], campaign.get("name", ""), status, client_id,
                        created_utc, created_ist, updated_utc, updated_ist,
                        "Yes" if included else "No"
                    ])
                    
                except Exception as e:
                    self.logger.error(f"Error processing campaign {campaign.get('id')}: {e}")
        
        self.logger.info(f"Filtered to {len(filtered)} campaigns, summary saved to {summary_csv}")
        self._log_audit("FILTER_CAMPAIGNS", {
            "total_campaigns": len(campaigns),
            "filtered_campaigns": len(filtered),
            "summary_file": summary_csv
        })
        
        return filtered

    async def export_campaign_leads(self, campaign_id: str) -> Optional[str]:
        """Export leads for a specific campaign"""
        export_file = f"{self.config.exports_dir}/leads_campaign_{campaign_id}.csv"
        
        # Remove stale export
        if os.path.exists(export_file):
            os.remove(export_file)
        
        url = f"{self.config.base_url}/campaigns/{campaign_id}/leads-export"
        response = await self._make_request("GET", url)
        
        if response and 'text/csv' in response.headers.get('Content-Type', ''):
            content = await response.read()
            with open(export_file, 'wb') as f:
                f.write(content)
            
            self.logger.info(f"Exported leads for campaign {campaign_id}")
            return export_file
        
        self.logger.error(f"Failed to export leads for campaign {campaign_id}")
        return None

    def filter_no_reply_leads(self, csv_file: str) -> Tuple[pd.DataFrame, int, int]:
        """Filter leads with no replies"""
        try:
            df = pd.read_csv(csv_file)
            total_leads = len(df)
            no_reply_leads = df[df['reply_count'] == 0]
            no_reply_count = len(no_reply_leads)
            
            self.logger.info(f"Found {no_reply_count}/{total_leads} no-reply leads in {csv_file}")
            return no_reply_leads, total_leads, no_reply_count
            
        except Exception as e:
            self.logger.error(f"Error filtering leads from {csv_file}: {e}")
            return pd.DataFrame(), 0, 0

    async def select_campaigns_for_processing(self, campaigns: List[dict]) -> List[dict]:
        """Dynamically select campaigns to reach target leads"""
        self.logger.info(f"Selecting campaigns to reach target of {self.config.target_leads} leads")
        
        # Get lead counts for all campaigns
        campaign_data = []
        
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        
        async def get_campaign_data(campaign):
            async with semaphore:
                campaign_id = campaign["id"]
                campaign_name = campaign.get("name", "")
                
                csv_file = await self.export_campaign_leads(campaign_id)
                if csv_file:
                    _, total_leads, no_reply_count = self.filter_no_reply_leads(csv_file)
                    return {
                        'campaign': campaign,
                        'total_leads': total_leads,
                        'no_reply_count': no_reply_count,
                        'csv_file': csv_file
                    }
                return None
        
        # Process campaigns concurrently
        tasks = [get_campaign_data(campaign) for campaign in campaigns]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        campaign_data = [r for r in results if r and not isinstance(r, Exception)]
        
        # Sort by no_reply_count descending
        campaign_data.sort(key=lambda x: x['no_reply_count'], reverse=True)
        
        # Select campaigns to reach target
        selected_campaigns = []
        cumulative_leads = 0
        
        for data in campaign_data:
            if data['no_reply_count'] <= 0:
                break
            
            selected_campaigns.append(data)
            cumulative_leads += data['no_reply_count']
            self.stats['campaigns_processed'] += 1
            
            if cumulative_leads >= self.config.target_leads:
                break
        
        self.logger.info(f"Selected {len(selected_campaigns)} campaigns with {cumulative_leads} total no-reply leads")
        self._log_audit("SELECT_CAMPAIGNS", {
            "selected_count": len(selected_campaigns),
            "total_no_reply_leads": cumulative_leads,
            "target": self.config.target_leads
        })
        
        return selected_campaigns

    def create_backup(self, leads_df: pd.DataFrame) -> str:
        """Create compressed backup of leads before deletion"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{self.config.backup_dir}/leads_backup_{timestamp}.csv"
        backup_gz = f"{backup_file}.gz"
        
        # Add metadata columns
        backup_df = leads_df.copy()
        backup_df['backup_timestamp'] = timestamp
        backup_df['deletion_reason'] = 'no_reply_cleanup'
        backup_df['target_leads'] = self.config.target_leads
        backup_df['days_threshold'] = self.config.days_threshold
        
        # Save and compress
        backup_df.to_csv(backup_file, index=False)
        
        with open(backup_file, 'rb') as f_in:
            with gzip.open(backup_gz, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        os.remove(backup_file)  # Remove uncompressed version
        
        self.stats['leads_backed_up'] = len(backup_df)
        self.logger.info(f"Created compressed backup: {backup_gz} ({len(backup_df)} leads)")
        self._log_audit("CREATE_BACKUP", {
            "backup_file": backup_gz,
            "leads_count": len(backup_df)
        })
        
        return backup_gz

    async def validate_lead_before_deletion(self, campaign_id: str, lead_id: str) -> bool:
        """Re-validate lead still meets deletion criteria"""
        url = f"{self.config.base_url}/campaigns/{campaign_id}/leads/{lead_id}"
        response = await self._make_request("GET", url)
        
        if response and response.status == 200:
            try:
                lead_data = await response.json()
                current_reply_count = lead_data.get('reply_count', 0)
                
                if current_reply_count > 0:
                    self.logger.warning(f"Lead {lead_id} now has {current_reply_count} replies, skipping deletion")
                    self.stats['validation_failures'] += 1
                    return False
                    
            except Exception as e:
                self.logger.error(f"Error validating lead {lead_id}: {e}")
                return False
        
        return True

    async def delete_lead(self, campaign_id: str, lead_id: str) -> bool:
        """Delete a single lead with validation"""
        if self.config.dry_run:
            self.logger.info(f"DRY-RUN: Would delete lead {lead_id} from campaign {campaign_id}")
            return True
        
        # Pre-deletion validation
        if not await self.validate_lead_before_deletion(campaign_id, lead_id):
            return False
        
        url = f"{self.config.base_url}/campaigns/{campaign_id}/leads/{lead_id}"
        response = await self._make_request("DELETE", url)
        
        if response and response.status in [200, 404]:
            self.logger.debug(f"Successfully deleted lead {lead_id} from campaign {campaign_id}")
            self._log_audit("DELETE_LEAD", {
                "lead_id": lead_id,
                "campaign_id": campaign_id,
                "success": True
            })
            return True
        
        self.logger.error(f"Failed to delete lead {lead_id} from campaign {campaign_id}")
        self._log_audit("DELETE_LEAD", {
            "lead_id": lead_id,
            "campaign_id": campaign_id,
            "success": False
        })
        return False

    async def process_deletion_batch(self, leads_batch: List[tuple]) -> List[bool]:
        """Process a batch of deletions concurrently"""
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        
        async def delete_with_semaphore(campaign_id, lead_id):
            async with semaphore:
                return await self.delete_lead(campaign_id, lead_id)
        
        tasks = [delete_with_semaphore(campaign_id, lead_id) for campaign_id, lead_id in leads_batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Batch deletion error: {result}")
                processed_results.append(False)
            else:
                processed_results.append(result)
        
        return processed_results

    def aggregate_no_reply_leads(self, selected_campaigns: List[dict]) -> pd.DataFrame:
        """Aggregate all no-reply leads into single DataFrame"""
        all_leads = []
        
        for campaign_data in selected_campaigns:
            campaign = campaign_data['campaign']
            csv_file = campaign_data['csv_file']
            
            no_reply_df, _, _ = self.filter_no_reply_leads(csv_file)
            
            if not no_reply_df.empty:
                no_reply_df['Campaign ID'] = campaign['id']
                no_reply_df['Campaign Name'] = campaign.get('name', '')
                all_leads.append(no_reply_df)
        
        if all_leads:
            aggregated_df = pd.concat(all_leads, ignore_index=True)
            self.stats['leads_exported'] = len(aggregated_df)
            
            # Save aggregated file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{self.config.exports_dir}/aggregated_no_reply_leads_{timestamp}.csv"
            aggregated_df.to_csv(output_file, index=False)
            
            self.logger.info(f"Aggregated {len(aggregated_df)} no-reply leads to {output_file}")
            return aggregated_df
        
        return pd.DataFrame()

    def confirm_deletion(self, total_leads: int) -> bool:
        """Get user confirmation for destructive operations"""
        if self.config.dry_run:
            return True
            
        print(f"\n⚠️  DANGER: About to delete {total_leads} leads permanently!")
        print("This action CANNOT be undone.")
        print(f"Backup will be created in: {self.config.backup_dir}")
        
        confirmation = input("Type 'DELETE_CONFIRMED' to proceed: ")
        if confirmation != 'DELETE_CONFIRMED':
            self.logger.info("Deletion cancelled by user")
            return False
        
        return True

    async def delete_leads(self, leads_df: pd.DataFrame) -> None:
        """Delete leads in batches with progress tracking"""
        total_leads = len(leads_df)
        batch_size = 50
        
        if not self.confirm_deletion(total_leads):
            return
        
        # Create backup
        backup_file = self.create_backup(leads_df)
        
        self.logger.info(f"Starting deletion of {total_leads} leads in batches of {batch_size}")
        
        # Prepare deletion list
        deletion_list = [(row['Campaign ID'], row['id']) for _, row in leads_df.iterrows()]
        
        # Process in batches
        for i in range(0, len(deletion_list), batch_size):
            batch = deletion_list[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(deletion_list) + batch_size - 1) // batch_size
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} leads)")
            
            results = await self.process_deletion_batch(batch)
            
            # Update statistics
            success_count = sum(1 for r in results if r)
            failure_count = len(results) - success_count
            
            self.stats['leads_deleted_success'] += success_count
            self.stats['leads_deleted_failed'] += failure_count
            
            # Progress update
            processed = i + len(batch)
            progress = (processed / total_leads) * 100
            elapsed = time.time() - self.start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total_leads - processed) / rate if rate > 0 else 0
            
            self.logger.info(f"Progress: {processed}/{total_leads} ({progress:.1f}%) "
                           f"Rate: {rate:.1f}/sec ETA: {eta:.0f}s")
            
            # Small delay between batches
            await asyncio.sleep(1)

    def save_audit_log(self) -> str:
        """Save audit trail to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audit_file = f"{self.config.logs_dir}/audit_log_{timestamp}.json"
        
        with open(audit_file, 'w') as f:
            json.dump(self.audit_log, f, indent=2)
        
        self.logger.info(f"Audit log saved to {audit_file}")
        return audit_file

    def send_email_report(self, backup_file: str, audit_file: str):
        """Send comprehensive email report"""
        if not self.config.sender_email or not self.config.recipient_emails:
            self.logger.warning("Email configuration missing, skipping email report")
            return
        
        try:
            # Calculate execution time
            execution_time = time.time() - self.start_time
            
            # Create email content
            subject = f"Smart Lead Cleanup Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            body = f"""
Smart Lead Cleanup Execution Report
================================

EXECUTION SUMMARY:
- Mode: {'DRY RUN' if self.config.dry_run else 'PRODUCTION'}
- Execution Time: {execution_time:.2f} seconds ({execution_time/60:.1f} minutes)
- Target Leads: {self.config.target_leads:,}
- Days Threshold: {self.config.days_threshold}

PROCESSING STATISTICS:
- Campaigns Processed: {self.stats['campaigns_processed']}
- Leads Exported: {self.stats['leads_exported']:,}
- Leads Backed Up: {self.stats['leads_backed_up']:,}
- Leads Successfully Deleted: {self.stats['leads_deleted_success']:,}
- Leads Failed to Delete: {self.stats['leads_deleted_failed']:,}
- API Errors: {self.stats['api_errors']}
- Validation Failures: {self.stats['validation_failures']}

SUCCESS RATE: {(self.stats['leads_deleted_success'] / max(1, self.stats['leads_deleted_success'] + self.stats['leads_deleted_failed']) * 100):.1f}%

BACKUP INFORMATION:
- Backup File: {os.path.basename(backup_file)}
- Location: {backup_file}

{'DRY RUN NOTICE: No actual deletions were performed.' if self.config.dry_run else ''}

For detailed logs, see attached audit file.
            """
            
            # Create email
            msg = EmailMessage()
            msg["From"] = self.config.sender_email
            msg["To"] = ", ".join(self.config.recipient_emails)
            msg["Subject"] = subject
            msg.set_content(body)
            
            # Attach backup file (if exists and not too large)
            if os.path.exists(backup_file) and os.path.getsize(backup_file) < 25 * 1024 * 1024:  # 25MB limit
                with open(backup_file, 'rb') as f:
                    backup_data = f.read()
                msg.add_attachment(backup_data, maintype='application', subtype='gzip',
                                 filename=os.path.basename(backup_file))
            
            # Attach audit log
            if os.path.exists(audit_file):
                with open(audit_file, 'rb') as f:
                    audit_data = f.read()
                msg.add_attachment(audit_data, maintype='application', subtype='json',
                                 filename=os.path.basename(audit_file))
            
            # Attach log file
            if os.path.exists(self.log_file):
                with open(self.log_file, 'rb') as f:
                    log_data = f.read()
                msg.add_attachment(log_data, maintype='text', subtype='plain',
                                 filename=os.path.basename(self.log_file))
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.config.smtp_server, self.config.smtp_port, context=context) as server:
                server.login(self.config.sender_email, self.config.smtp_password)
                server.send_message(msg)
            
            self.logger.info("Email report sent successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to send email report: {e}")

    async def run(self):
        """Main execution flow"""
        self.start_time = time.time()
        self.logger.info(f"Starting Smart Lead Cleanup (Target: {self.config.target_leads}, Days: {self.config.days_threshold}, Dry-Run: {self.config.dry_run})")
        
        try:
            # Phase 1: Campaign Discovery
            campaigns = await self.fetch_campaigns()
            if not campaigns:
                self.logger.error("No campaigns found, exiting")
                return
            
            # Phase 2: Campaign Filtering
            filtered_campaigns = self.filter_campaigns(campaigns)
            if not filtered_campaigns:
                self.logger.error("No campaigns match filtering criteria, exiting")
                return
            
            # Phase 3: Campaign Selection & Lead Export
            selected_campaigns = await self.select_campaigns_for_processing(filtered_campaigns)
            if not selected_campaigns:
                self.logger.error("No campaigns selected for processing, exiting")
                return
            
            # Phase 4: Lead Aggregation
            aggregated_leads = self.aggregate_no_reply_leads(selected_campaigns)
            if aggregated_leads.empty:
                self.logger.info("No no-reply leads found, exiting")
                return
            
            # Phase 5: Lead Deletion
            await self.delete_leads(aggregated_leads)
            
        except Exception as e:
            self.logger.error(f"Unexpected error during execution: {e}")
            raise
        
        finally:
            # Save audit log and send report
            audit_file = self.save_audit_log()
            backup_file = ""  # Will be set if backup was created
            
            # Find the most recent backup file
            backup_files = list(Path(self.config.backup_dir).glob("leads_backup_*.csv.gz"))
            if backup_files:
                backup_file = str(max(backup_files, key=lambda p: p.stat().st_mtime))
            
            self.send_email_report(backup_file, audit_file)
            
            execution_time = time.time() - self.start_time
            self.logger.info(f"Smart Lead Cleanup completed in {execution_time:.2f} seconds")
            self.logger.info(f"Final stats: {self.stats}")

def load_config() -> Config:
    """Load configuration from environment variables"""
    api_key = os.getenv('SMARTLEAD_API_KEY')
    if not api_key:
        raise ValueError("SMARTLEAD_API_KEY environment variable is required")
    
    # Parse exclude client IDs
    exclude_ids_str = os.getenv('EXCLUDE_CLIENT_IDS', '1598')
    exclude_client_ids = set(int(x.strip()) for x in exclude_ids_str.split(',') if x.strip())
    
    # Parse recipient emails
    recipients_str = os.getenv('RECIPIENT_EMAILS', '')
    recipient_emails = [email.strip() for email in recipients_str.split(',') if email.strip()]
    
    return Config(
        api_key=api_key,
        target_leads=int(os.getenv('TARGET_LEADS', 30000)),
        days_threshold=int(os.getenv('DAYS_THRESHOLD', 45)),
        max_concurrency=int(os.getenv('MAX_CONCURRENCY', 10)),
        exclude_client_ids=exclude_client_ids,
        sender_email=os.getenv('SENDER_EMAIL', ''),
        smtp_password=os.getenv('SMTP_PASSWORD', ''),
        recipient_emails=recipient_emails,
        backup_dir=os.getenv('BACKUP_DIR', 'backups'),
        logs_dir=os.getenv('LOGS_DIR', 'logs'),
        exports_dir=os.getenv('EXPORTS_DIR', 'exports')
    )

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Smart Lead Cleanup - Safely delete inactive leads')
    parser.add_argument('--dry-run', action='store_true', help='Simulate deletion without actually deleting')
    parser.add_argument('--target-leads', type=int, help='Target number of leads to delete')
    parser.add_argument('--days-threshold', type=int, help='Minimum days since campaign creation')
    parser.add_argument('--max-concurrency', type=int, help='Maximum concurrent API requests')
    
    args = parser.parse_args()
    
    try:
        config = load_config()
        
        # Override with command line arguments
        if args.dry_run:
            config.dry_run = True
        if args.target_leads:
            config.target_leads = args.target_leads
        if args.days_threshold:
            config.days_threshold = args.days_threshold
        if args.max_concurrency:
            config.max_concurrency = args.max_concurrency
        
        # Run the cleanup
        async def run_cleanup():
            async with SmartLeadCleanup(config) as cleanup:
                await cleanup.run()
        
        asyncio.run(run_cleanup())
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()