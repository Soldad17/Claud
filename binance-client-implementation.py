# backend/brokers/binance_client.py
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
import pandas as pd
import numpy as np
import time
import logging
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional, Union, Tuple

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class BinanceClient:
    def __init__(self, api_key=None, api_secret=None, simulado=False):
        """
        Initialize Binance client
        
        Parameters:
        - api_key: Binance API key
        - api_secret: Binance API secret
        - simulado: If True, run in simulation mode
        """
        self.simulado = simulado
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        
        if not self.simulado and (not self.api_key or not self.api_secret):
            logger.warning("API key or secret not provided. Using simulation mode.")
            self.simulado = True
            
        if not self.simulado:
            try:
                self.client = Client(self.api_key, self.api_secret)
                # Test connection
                self.client.get_account()
                logger.info("Successfully connected to Binance API")
            except (BinanceAPIException, BinanceRequestException) as e:
                logger.error(f"Failed to connect to Binance API: {e}")
                logger.warning("Falling back to simulation mode")
                self.simulado = True
    
    def get_historical_klines(self, symbol: str, interval: str, 
                              start_str: str, end_str: Optional[str] = None) -> pd.DataFrame:
        """
        Get historical klines (candlestick data)
        
        Parameters:
        - symbol: Trading pair (e.g. 'BTCUSDT')
        - interval: Kline interval (e.g. '1d', '4h', '1h', '15m')
        - start_str: Start date string (e.g. '1 day ago', '1 Jan 2020')
        - end_str: End date string
        
        Returns:
        - DataFrame with OHLCV data
        """
        try:
            if not self.simulado:
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=start_str,
                    end_str=end_str
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Convert types
                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                                   'quote_asset_volume', 'taker_buy_base_asset_volume', 
                                   'taker_buy_quote_asset_volume']
                for column in numeric_columns:
                    df[column] = pd.to_numeric(df[column])
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                return df
            else:
                # Simulate data in simulation mode
                return self._simulate_historical_data(symbol, interval, start_str, end_str)
        except Exception as e:
            logger.error(f"Error getting historical klines: {e}")
            raise
    
    def _simulate_historical_data(self, symbol: str, interval: str, 
                                 start_str: str, end_str: Optional[str] = None) -> pd.DataFrame:
        """Generate simulated historical data for testing"""
        # Parse start and end dates
        if isinstance(start_str, str):
            if start_str.endswith('ago'):
                # Parse '1 day ago', '1 month ago', etc.
                parts = start_str.split()
                value = int(parts[0])
                unit = parts[1].lower()
                
                if unit in ['day', 'days']:
                    periods = value
                elif unit in ['hour', 'hours']:
                    periods = value // 24
                elif unit in ['month', 'months']:
                    periods = value * 30
                else:
                    periods = 30  # Default to 30 days
                    
                dates = pd.date_range(end='now', periods=periods)
            else:
                # Try to parse as date string
                try:
                    start_date = pd.to_datetime(start_str)
                    end_date = pd.to_datetime(end_str) if end_str else pd.Timestamp.now()
                    dates = pd.date_range(start=start_date, end=end_date)
                except:
                    # Fallback to 30 days
                    dates = pd.date_range(end='now', periods=30)
        else:
            # Fallback to 30 days
            dates = pd.date_range(end='now', periods=30)
            
        # Generate random price data with some trend and volatility
        np.random.seed(42)  # For reproducibility
        
        # Start with a base price that depends on the symbol
        if 'BTC' in symbol:
            base_price = 40000
        elif 'ETH' in symbol:
            base_price = 2000
        else:
            base_price = 100
            
        # Generate price series with trend and volatility
        price_series = []
        current_price = base_price
        
        for _ in range(len(dates)):
            # Add random walk with drift
            drift = np.random.normal(0.0001, 0.002)  # Small upward drift
            volatility = np.random.normal(0, 0.01)  # Random volatility
            current_price *= (1 + drift + volatility)
            
            # Create OHLC data
            daily_volatility = current_price * np.random.uniform(0.005, 0.02)
            open_price = current_price * (1 + np.random.normal(0, 0.003))
            high_price = max(open_price, current_price) + daily_volatility
            low_price = min(open_price, current_price) - daily_volatility
            close_price = current_price
            volume = np.random.uniform(100, 1000) * current_price
            
            price_series.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'quote_asset_volume': volume * close_price,
                'number_of_trades': int(np.random.uniform(1000, 5000)),
                'taker_buy_base_asset_volume': volume * 0.4,
                'taker_buy_quote_asset_volume': volume * close_price * 0.4
            })
            
        # Create DataFrame
        df = pd.DataFrame(price_series, index=dates)
        return df
    
    def get_account_info(self) -> Dict:
        """Get account information including balances"""
        if not self.simulado:
            try:
                account_info = self.client.get_account()
                return account_info
            except Exception as e:
                logger.error(f"Error getting account info: {e}")
                raise
        else:
            # Return simulated account info
            return {
                'makerCommission': 10,
                'takerCommission': 10,
                'buyerCommission': 0,
                'sellerCommission': 0,
                'canTrade': True,
                'canWithdraw': True,
                'canDeposit': True,
                'balances': [
                    {
                        'asset': 'BTC',
                        'free': '0.5',
                        'locked': '0'
                    },
                    {
                        'asset': 'ETH',
                        'free': '10',
                        'locked': '0'
                    },
                    {
                        'asset': 'USDT',
                        'free': '10000',
                        'locked': '0'
                    }
                ]
            }
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book for a symbol"""
        if not self.simulado:
            try:
                order_book = self.client.get_order_book(symbol=symbol, limit=limit)
                return order_book
            except Exception as e:
                logger.error(f"Error getting order book: {e}")
                raise
        else:
            # Generate simulated order book
            current_price = 0
            
            if 'BTC' in symbol:
                current_price = 40000
            elif 'ETH' in symbol:
                current_price = 2000
            else:
                current_price = 100
                
            # Generate bids (buy orders)
            bids = []
            for i in range(limit):
                price = current_price * (1 - 0.0001 * i)
                quantity = np.random.uniform(0.1, 2.0)
                bids.append([str(price), str(quantity)])
                
            # Generate asks (sell orders)
            asks = []
            for i in range(limit):
                price = current_price * (1 + 0.0001 * i)
                quantity = np.random.uniform(0.1, 2.0)
                asks.append([str(price), str(quantity)])
                
            return {
                'lastUpdateId': int(time.time() * 1000),
                'bids': bids,
                'asks': asks
            }
    
    def execute_order(self, symbol: str, side: str, order_type: str, 
                     quantity: float, price: Optional[float] = None) -> Dict:
        """
        Execute a trading order
        
        Parameters:
        - symbol: Trading pair (e.g. 'BTCUSDT')
        - side: 'BUY' or 'SELL'
        - order_type: 'MARKET', 'LIMIT', etc.
        - quantity: Amount to buy/sell
        - price: Limit price (required for LIMIT orders)
        
        Returns:
        - Order information
        """
        if not self.simulado:
            try:
                params = {
                    'symbol': symbol,
                    'side': side,
                    'type': order_type,
                    'quantity': quantity
                }
                
                if order_type == 'LIMIT':
                    if price is None:
                        raise ValueError("Price must be specified for LIMIT orders")
                    params['price'] = price
                    params['timeInForce'] = 'GTC'  # Good Till Cancelled
                
                # Execute order
                response = self.client.create_order(**params)
                logger.info(f"Order executed: {symbol} {side} {quantity} @ {price}")
                return response
            except Exception as e:
                logger.error(f"Error executing order: {e}")
                raise
        else:
            # Simulate order execution
            current_price = 0
            if 'BTC' in symbol:
                current_price = 40000
            elif 'ETH' in symbol:
                current_price = 2000
            else:
                current_price = 100
                
            exec_price = price if price and order_type == 'LIMIT' else current_price
            
            # Simulate order fill (might be partial)
            fill_percent = np.random.uniform(0.95, 1.0)
            filled_qty = quantity * fill_percent
            
            order_id = f"sim_{int(time.time() * 1000)}"
            
            response = {
                'symbol': symbol,
                'orderId': order_id,
                'clientOrderId': f'simulated_{order_id}',
                'transactTime': int(time.time() * 1000),
                'price': str(exec_price),
                'origQty': str(quantity),
                'executedQty': str(filled_qty),
                'status': 'FILLED' if fill_percent >= 0.99 else 'PARTIALLY_FILLED',
                'timeInForce': 'GTC' if order_type == 'LIMIT' else 'GTT',
                'type': order_type,
                'side': side,
                'fills': [
                    {
                        'price': str(exec_price),
                        'qty': str(filled_qty),
                        'commission': str(filled_qty * exec_price * 0.001),
                        'commissionAsset': symbol.replace('USDT', '')
                    }
                ]
            }
            
            logger.info(f"Simulated order executed: {symbol} {side} {filled_qty}/{quantity} @ {exec_price}")
            return response
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open orders for a symbol or all symbols"""
        if not self.simulado:
            try:
                if symbol:
                    open_orders = self.client.get_open_orders(symbol=symbol)
                else:
                    open_orders = self.client.get_open_orders()
                return open_orders
            except Exception as e:
                logger.error(f"Error getting open orders: {e}")
                raise
        else:
            # Return simulated open orders (empty list for simplicity)
            return []
    
    def cancel_order(self, symbol: str, order_id: Union[str, int]) -> Dict:
        """Cancel an open order"""
        if not self.simulado:
            try:
                result = self.client.cancel_order(symbol=symbol, orderId=order_id)
                return result
            except Exception as e:
                logger.error(f"Error cancelling order: {e}")
                raise
        else:
            # Simulate cancel order response
            return {
                'symbol': symbol,
                'origClientOrderId': f'simulated_{order_id}',
                'orderId': order_id,
                'status': 'CANCELED'
            }
    
    def get_ticker_price(self, symbol: Optional[str] = None) -> Union[Dict, List[Dict]]:
        """Get latest price for a symbol or all symbols"""
        if not self.simulado:
            try:
                if symbol:
                    price = self.client.get_symbol_ticker(symbol=symbol)
                else:
                    price = self.client.get_all_tickers()
                return price
            except Exception as e:
                logger.error(f"Error getting ticker price: {e}")
                raise
        else:
            # Simulate price data
            if symbol:
                if 'BTC' in symbol:
                    price = 40000 * (1 + np.random.normal(0, 0.001))
                elif 'ETH' in symbol:
                    price = 2000 * (1 + np.random.normal(0, 0.001))
                else:
                    price = 100 * (1 + np.random.normal(0, 0.001))
                    
                return {
                    'symbol': symbol,
                    'price': str(price)
                }
            else:
                # Return prices for common pairs
                common_pairs = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT']
                prices = []
                
                for pair in common_pairs:
                    if 'BTC' in pair:
                        price = 40000 * (1 + np.random.normal(0, 0.001))
                    elif 'ETH' in pair:
                        price = 2000 * (1 + np.random.normal(0, 0.001))
                    elif 'ADA' in pair:
                        price = 1.5 * (1 + np.random.normal(0, 0.001))
                    elif 'DOGE' in pair:
                        price = 0.30 * (1 + np.random.normal(0, 0.001))
                    elif 'XRP' in pair:
                        price = 0.75 * (1 + np.random.normal(0, 0.001))
                    else:
                        price = 100 * (1 + np.random.normal(0, 0.001))
                        
                    prices.append({
                        'symbol': pair,
                        'price': str(price)
                    })
                    
                return prices
