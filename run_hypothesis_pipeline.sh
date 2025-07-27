#!/bin/bash

# Automated PlaceboRx Hypothesis Testing Pipeline
# Shell script wrapper for easy execution

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if we're in the right directory
    if [[ ! -f "package.json" ]] || [[ ! -f "vercel.json" ]]; then
        print_error "Must run from the project root directory"
        exit 1
    fi
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check if git is available
    if ! command -v git &> /dev/null; then
        print_error "Git is required but not installed"
        exit 1
    fi
    
    # Check if we're in a git repository
    if [[ ! -d ".git" ]]; then
        print_error "Not in a git repository"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Install requirements
    print_status "Installing Python requirements..."
    if [[ -f "requirements_enhanced.txt" ]]; then
        pip install -r requirements_enhanced.txt
    elif [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
    else
        print_warning "No requirements file found, installing basic packages..."
        pip install pandas requests numpy matplotlib seaborn
    fi
    
    print_success "Environment setup completed"
}

# Function to run the pipeline
run_pipeline() {
    print_status "Starting automated hypothesis testing pipeline..."
    
    # Check if the automation script exists
    if [[ ! -f "automated_hypothesis_pipeline.py" ]]; then
        print_error "Automation script not found: automated_hypothesis_pipeline.py"
        exit 1
    fi
    
    # Run the Python automation script
    python3 automated_hypothesis_pipeline.py
    
    if [[ $? -eq 0 ]]; then
        print_success "Pipeline completed successfully!"
    else
        print_error "Pipeline failed!"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -s, --setup-only    Only setup environment, don't run pipeline"
    echo "  -c, --check-only    Only check prerequisites, don't run pipeline"
    echo "  -v, --verbose       Enable verbose output"
    echo ""
    echo "Examples:"
    echo "  $0                  # Run complete pipeline"
    echo "  $0 --setup-only     # Only setup environment"
    echo "  $0 --check-only     # Only check prerequisites"
}

# Main execution
main() {
    echo "🚀 PlaceboRx Automated Hypothesis Testing Pipeline"
    echo "=================================================="
    
    # Parse command line arguments
    SETUP_ONLY=false
    CHECK_ONLY=false
    VERBOSE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -s|--setup-only)
                SETUP_ONLY=true
                shift
                ;;
            -c|--check-only)
                CHECK_ONLY=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Check prerequisites
    check_prerequisites
    
    if [[ "$CHECK_ONLY" == true ]]; then
        print_success "Prerequisites check completed"
        exit 0
    fi
    
    # Setup environment
    setup_environment
    
    if [[ "$SETUP_ONLY" == true ]]; then
        print_success "Environment setup completed"
        exit 0
    fi
    
    # Run the pipeline
    run_pipeline
    
    print_success "All done! 🎉"
}

# Run main function
main "$@" 