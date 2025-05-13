from datetime import datetime

def get_most_recent_contract(contracts):
    """
    From a list of contract details for NQ futures, select the contract with the earliest expiration date
    that is still in the future (the active or front-month contract).
    """
    # Helper function to parse the expiration string.
    def parse_date(date_str):
        try:
            # Handle format 'YYYYMMDD' or sometimes 'YYYYMM'
            if len(date_str) == 8:
                return datetime.strptime(date_str, '%Y%m%d')
            elif len(date_str) == 6:
                return datetime.strptime(date_str, '%Y%m')
        except Exception as e:
            return datetime.max

    today = datetime.today()
    # Filter contracts that have an expiration date in the future.
    valid_contracts = [cd for cd in contracts if parse_date(cd.contract.lastTradeDateOrContractMonth) > today]
    if valid_contracts:
        # Sort filtered contracts by expiration date (earliest first)
        sorted_contracts = sorted(valid_contracts, key=lambda cd: parse_date(cd.contract.lastTradeDateOrContractMonth))
        return sorted_contracts[0].contract
    else:
        # Fallback to the first contract if for some reason none are in the future.
        return contracts[0].contract 