# backend/api/main.py
from fastapi import FastAPI, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.security.api_key import APIKeyHeader
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
import logging
import os
import sys
import time
from dotenv import load_dotenv

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import app modules
from models.lstm_model import LSTMModel
from brokers.binance_client import BinanceClient
# Import other modules as needed

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Sistema de Trading API",
    description="API para o sistema de trading com IA",
    version="4.0.0"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "development_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# API Key security for automated access
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# OAuth2 for user authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Sample API keys (in production, use a database)
API_KEYS = {
    os.getenv("SAMPLE_API_KEY", "test_api_key"): "admin"
}

# Sample users (in production, use a database with hashed passwords)
USERS = {
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "email": "admin@example.com",
        "hashed_password": "password123",  # In production, use hashed passwords
        "disabled": False,
    }
}

# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class TradeRequest(BaseModel):
    symbol: str
    side: str
    quantity: float
    price: Optional[float] = None
    type: str = "MARKET"

class PredictionRequest(BaseModel):
    symbol: str
    interval: str = "1h"
    lookback: str = "30 days"

# Security functions
def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    # In production, verify password with hashing
    if password != user.hashed_password:
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(USERS, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header in API_KEYS:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, detail="Could not validate API key"
    )

# Global instances
lstm_model = LSTMModel()
binance_client = BinanceClient(simulado=True)  # Default to simulation mode

# Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request to {request.url.path} took {process_time:.3f}s")
    return response

# Route for login
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(USERS, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# API routes
@app.get("/")
async def read_root():
    """Endpoint raiz da API"""
    return {"message": "Sistema de Trading Online API v4.0.0"}

@app.get("/status")
async def get_status(current_user: User = Depends(get_current_active_user)):
    """Verificar status do sistema"""
    return {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "user": current_user.username,
        "simulation_mode": binance_client.simulado
    }

@app.post("/api/predict", response_model=Dict[str, Any])
async def predict_prices(
    request: PredictionRequest,
    api_key: str = Depends(get_api_key)
):
    """Make price predictions using the LSTM model"""
    try:
        # Get historical data
        klines = binance_client.get_historical_klines(
            symbol=request.symbol,
            interval=request.interval,
            start_str=request.lookback
        )
        
        # Extract close prices
        closes = klines['close'].values
        
        # Get trading signals
        signal, predicted_price, percent_change = lstm_model.get_trading_signals(closes)
        
        return {
            "symbol": request.symbol,
            "signal": signal,
            "current_price": closes[-1],
            "predicted_price": float(predicted_price),
            "percent_change": float(percent_change),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )

@app.post("/api/trade", response_model=Dict[str, Any])
async def execute_trade(
    request: TradeRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """Execute a trade order"""
    try:
        # Execute order
        result = binance_client.execute_order(
            symbol=request.symbol,
            side=request.side,
            order_type=request.type,
            quantity=request.quantity,
            price=request.price
        )
        
        # Log trade in background
        background_tasks.add_task(log_trade, current_user.username, request, result)
        
        return {
            "status": "success",
            "order": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing trade: {str(e)}"
        )

@app.get("/api/account")
async def get_account_info(current_user: User = Depends(get_current_active_user)):
    """Get account information"""
    try:
        account_info = binance_client.get_account_info()
        return account_info
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting account info: {str(e)}"
        )

@app.get("/api/market/{symbol}")
async def get_market_data(
    symbol: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get current market data for a symbol"""
    try:
        # Get current price
        ticker = binance_client.get_ticker_price(symbol=symbol)
        
        # Get order book
        order_book = binance_client.get_order_book(symbol=symbol, limit=10)
        
        return {
            "symbol": symbol,
            "price": ticker["price"],
            "order_book": {
                "bids": order_book["bids"][:5],
                "asks": order_book["asks"][:5]
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting market data: {str(e)}"
        )

# Utility functions
def log_trade(username: str, request: TradeRequest, result: Dict):
    """Log trade details (in production, store in database)"""
    logger.info(f"Trade executed by {username}: {request.symbol} {request.side} {request.quantity} {result}")
    # In production, store trade in database

# Main entry point
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    
    # Load model if available
    try:
        lstm_model.load()
        logger.info("LSTM model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load LSTM model: {e}")
        logger.info("Creating new LSTM model")
        lstm_model.build_model()
        
    # Run server
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
