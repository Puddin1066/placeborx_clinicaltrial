#!/usr/bin/env python3
"""
Repository Cleanup Script for PlaceboRx
Identifies and archives non-critical files to streamline the pipeline
"""

import os
import shutil
import json
from datetime import datetime
from typing import List, Dict, Set

class RepositoryCleanup:
    """Clean up non-critical files from the PlaceboRx repository"""
    
    def __init__(self):
        self.critical_files = self._get_critical_files()
        self.backup_dir = f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def _get_critical_files(self) -> Set[str]:
        """Define which files are critical for the pipeline"""
        
        # Core pipeline files
        core_pipeline = {
            'automated_hypothesis_pipeline.py',  # Main automation script
            'run_hypothesis_pipeline.sh',        # Shell wrapper
            'enhanced_main_pipeline.py',         # Enhanced pipeline
            'main_pipeline.py',                  # Basic pipeline
            'config.py',                         # Configuration
            'enhanced_config.py',                # Enhanced config
        }
        
        # Data analysis modules
        analysis_modules = {
            'clinical_trials_analyzer.py',       # Clinical trials analysis
            'market_analyzer.py',                # Market analysis
            'pubmed_analyzer.py',                # Literature analysis
            'openai_processor.py',               # AI processing
            'data_quality.py',                   # Data validation
            'ml_enhancement.py',                 # ML enhancement
            'visualization_engine.py',           # Visualization
        }
        
        # Frontend files (Vercel deployment)
        frontend_files = {
            'package.json',                      # Node.js dependencies
            'next.config.js',                    # Next.js config
            'vercel.json',                       # Vercel deployment
            'pages/',                            # Next.js pages
            'styles/',                           # CSS styles
            'tailwind.config.js',                # Tailwind CSS
            'postcss.config.js',                 # PostCSS config
        }
        
        # Documentation and guides
        docs = {
            'AUTOMATION_README.md',              # Main documentation
            'DATA_AUTHENTICITY_GUIDE.md',        # Data authenticity guide
            'check_data_authenticity.py',        # Data authenticity checker
        }
        
        # Requirements and environment
        requirements = {
            'requirements_enhanced.txt',         # Enhanced requirements
            'requirements.txt',                  # Basic requirements
            'requirements_minimal.txt',          # Minimal requirements
            'env_example.txt',                   # Environment example
            '.env',                              # Environment variables
            '.gitignore',                        # Git ignore
        }
        
        # GitHub Actions
        github_actions = {
            '.github/',                          # GitHub workflows
        }
        
        # Test files (keep essential ones)
        test_files = {
            'test_automation.py',                # Automation tests
        }
        
        # Content files
        content_files = {
            'vercel_landing_content.js',         # Landing page content
        }
        
        # Combine all critical files
        critical = (core_pipeline | analysis_modules | frontend_files | 
                   docs | requirements | github_actions | test_files | content_files)
        
        return critical
    
    def _get_non_critical_files(self) -> List[str]:
        """Identify files that are not critical for the pipeline"""
        
        non_critical = []
        
        # Files to archive (non-critical)
        archive_candidates = [
            # Backup files
            'vercel_landing_content.js.backup.*',
            
            # Generated data files (can be regenerated)
            '*.csv',
            '*.json',
            '*.md',
            
            # Log files
            '*.log',
            
            # Test files (except essential ones)
            'test_*.py',
            'simple_*.py',
            'working_*.py',
            
            # Alternative/experimental modules
            'olp_*.py',
            'comprehensive_*.py',
            'experimental_*.py',
            'hypothesis_testing_*.py',
            'enhanced_validation_*.py',
            'real_world_evidence_*.py',
            
            # Streamlit app (separate from main pipeline)
            'streamlit_app.py',
            'requirements_streamlit.txt',
            
            # Data authenticity reports (generated)
            'data_authenticity_report.md',
            
            # Python cache
            '__pycache__/',
            
            # Virtual environment (can be recreated)
            'venv/',
            
            # Vercel cache
            '.vercel/',
        ]
        
        # Check each file/directory
        for item in os.listdir('.'):
            if os.path.isfile(item):
                if item not in self.critical_files:
                    non_critical.append(item)
            elif os.path.isdir(item):
                if item not in self.critical_files and item not in ['.git']:
                    non_critical.append(item)
        
        return non_critical
    
    def _should_archive_file(self, filename: str) -> bool:
        """Determine if a file should be archived"""
        
        # Never archive these
        never_archive = {
            '.git', '.env', '.gitignore', 'README.md'
        }
        
        if filename in never_archive:
            return False
        
        # Archive these patterns
        archive_patterns = [
            # Backup files
            filename.startswith('vercel_landing_content.js.backup'),
            
            # Generated data files
            filename.endswith('.csv') and filename != 'clinical_trials_results.csv',
            filename.endswith('.json') and not filename.startswith('package'),
            filename.endswith('.md') and filename not in ['AUTOMATION_README.md', 'DATA_AUTHENTICITY_GUIDE.md'],
            
            # Log files
            filename.endswith('.log'),
            
            # Test files (except essential)
            filename.startswith('test_') and filename != 'test_automation.py',
            filename.startswith('simple_'),
            filename.startswith('working_'),
            
            # Alternative modules
            filename.startswith('olp_'),
            filename.startswith('comprehensive_'),
            filename.startswith('experimental_'),
            filename.startswith('hypothesis_testing_'),
            filename.startswith('enhanced_validation_'),
            filename.startswith('real_world_evidence_'),
            
            # Streamlit
            filename == 'streamlit_app.py',
            filename == 'requirements_streamlit.txt',
            
            # Generated reports
            filename == 'data_authenticity_report.md',
        ]
        
        return any(archive_patterns)
    
    def _should_archive_directory(self, dirname: str) -> bool:
        """Determine if a directory should be archived"""
        
        # Never archive these directories
        never_archive_dirs = {
            '.git', 'pages', 'styles', '.github'
        }
        
        if dirname in never_archive_dirs:
            return False
        
        # Archive these directories
        archive_dirs = {
            '__pycache__', 'venv', '.vercel'
        }
        
        return dirname in archive_dirs
    
    def create_archive(self) -> str:
        """Create archive of non-critical files"""
        
        print("🧹 PlaceboRx Repository Cleanup")
        print("=" * 50)
        
        # Create backup directory
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
        
        archived_files = []
        archived_dirs = []
        
        # Archive files
        for item in os.listdir('.'):
            item_path = os.path.join('.', item)
            
            if os.path.isfile(item_path):
                if self._should_archive_file(item):
                    dest_path = os.path.join(self.backup_dir, item)
                    shutil.move(item_path, dest_path)
                    archived_files.append(item)
                    print(f"📦 Archived file: {item}")
            
            elif os.path.isdir(item_path):
                if self._should_archive_directory(item):
                    dest_path = os.path.join(self.backup_dir, item)
                    shutil.move(item_path, dest_path)
                    archived_dirs.append(item)
                    print(f"📦 Archived directory: {item}")
        
        # Create archive manifest
        manifest = {
            'archive_created': datetime.now().isoformat(),
            'archived_files': archived_files,
            'archived_directories': archived_dirs,
            'total_items': len(archived_files) + len(archived_dirs)
        }
        
        manifest_path = os.path.join(self.backup_dir, 'archive_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✅ Cleanup completed!")
        print(f"📁 Archive created: {self.backup_dir}")
        print(f"📦 Archived {len(archived_files)} files and {len(archived_dirs)} directories")
        
        return self.backup_dir
    
    def show_critical_files(self):
        """Show which files are considered critical"""
        
        print("🔧 Critical Files (Kept):")
        print("=" * 50)
        
        categories = {
            'Core Pipeline': [
                'automated_hypothesis_pipeline.py',
                'run_hypothesis_pipeline.sh',
                'enhanced_main_pipeline.py',
                'main_pipeline.py',
                'config.py',
                'enhanced_config.py',
            ],
            'Analysis Modules': [
                'clinical_trials_analyzer.py',
                'market_analyzer.py',
                'pubmed_analyzer.py',
                'openai_processor.py',
                'data_quality.py',
                'ml_enhancement.py',
                'visualization_engine.py',
            ],
            'Frontend': [
                'package.json',
                'next.config.js',
                'vercel.json',
                'pages/',
                'styles/',
                'tailwind.config.js',
                'postcss.config.js',
            ],
            'Documentation': [
                'AUTOMATION_README.md',
                'DATA_AUTHENTICITY_GUIDE.md',
                'check_data_authenticity.py',
            ],
            'Requirements': [
                'requirements_enhanced.txt',
                'requirements.txt',
                'requirements_minimal.txt',
                'env_example.txt',
                '.env',
                '.gitignore',
            ],
            'GitHub Actions': [
                '.github/',
            ],
            'Tests': [
                'test_automation.py',
            ],
            'Content': [
                'vercel_landing_content.js',
            ]
        }
        
        for category, files in categories.items():
            print(f"\n{category}:")
            for file in files:
                if file in self.critical_files:
                    print(f"  ✅ {file}")
                else:
                    print(f"  ❌ {file} (not found)")
    
    def show_archive_contents(self):
        """Show what would be archived"""
        
        print("📦 Files to Archive:")
        print("=" * 50)
        
        for item in os.listdir('.'):
            if os.path.isfile(item):
                if self._should_archive_file(item):
                    print(f"📄 {item}")
            elif os.path.isdir(item):
                if self._should_archive_directory(item):
                    print(f"📁 {item}/")

def main():
    """Main execution"""
    cleanup = RepositoryCleanup()
    
    print("Choose an option:")
    print("1. Show critical files (preview)")
    print("2. Show files to archive (preview)")
    print("3. Create archive (cleanup)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        cleanup.show_critical_files()
    elif choice == '2':
        cleanup.show_archive_contents()
    elif choice == '3':
        archive_dir = cleanup.create_archive()
        print(f"\n🎉 Repository cleaned! Archive saved to: {archive_dir}")
        print("\nTo restore files later, run:")
        print(f"mv {archive_dir}/* .")
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main() 