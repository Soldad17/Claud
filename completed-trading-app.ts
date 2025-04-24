// App.tsx
import React, { useState, useEffect } from 'react';
import { 
  View, Text, StyleSheet, TextInput, TouchableOpacity, 
  ScrollView, SafeAreaView, ActivityIndicator, RefreshControl,
  Alert, StatusBar
} from 'react-native';
import axios from 'axios';
import { LineChart } from 'react-native-chart-kit';
import { Dimensions } from 'react-native';

const API_BASE_URL = 'http://localhost:8000'; // Mudar para o endereço real do servidor

interface Balance {
  asset: string;
  free: string;
  locked: string;
}

interface AccountInfo {
  balances: Balance[];
}

interface PredictionData {
  symbol: string;
  signal: string;
  current_price: number;
  predicted_price: number;
  percent_change: number;
  timestamp: string;
}

interface MarketData {
  timestamp: string;
  close: number;
}

const App = () => {
  const [authToken, setAuthToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [symbol, setSymbol] = useState<string>('BTCUSDT');
  const [accountInfo, setAccountInfo] = useState<AccountInfo | null>(null);
  const [predictions, setPredictions] = useState<PredictionData | null>(null);
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [orderAmount, setOrderAmount] = useState<string>('0.001');
  const [orderMessage, setOrderMessage] = useState<string | null>(null);
  const [orderStatus, setOrderStatus] = useState<'success' | 'error' | 'pending' | null>(null);
  
  // Telas do aplicativo
  const [currentScreen, setCurrentScreen] = useState<'login' | 'dashboard'>('login');
  
  // Autenticação
  const [username, setUsername] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const [loginError, setLoginError] = useState<string | null>(null);

  const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
      'Authorization': `Bearer ${authToken}`
    }
  });

  useEffect(() => {
    if (authToken) {
      fetchData();
    }
  }, [authToken, symbol]);

  const fetchData = async () => {
    setIsLoading(true);
    try {
      const [accountResponse, priceResponse] = await Promise.all([
        apiClient.get('/api/account'),
        apiClient.get(`/api/market/${symbol}`)
      ]);
      
      setAccountInfo(accountResponse.data);
      
      // Simular dados históricos para o gráfico
      const priceHistory = [];
      const currentPrice = parseFloat(priceResponse.data.price);
      const now = new Date();
      
      for (let i = 24; i >= 0; i--) {
        const time = new Date(now);
        time.setHours(time.getHours() - i);
        
        // Simular variação de preço
        const randomFactor = 0.98 + Math.random() * 0.04; // ±2%
        priceHistory.push({
          timestamp: time.toISOString(),
          close: currentPrice * randomFactor
        });
      }
      
      setMarketData(priceHistory);
      
      // Buscar previsões
      const predictionResponse = await apiClient.post('/api/predict', {
        symbol: symbol,
        interval: '1h',
        lookback: '30 days'
      });
      
      setPredictions(predictionResponse.data);
    } catch (error) {
      console.error('Erro ao buscar dados:', error);
      if (axios.isAxiosError(error) && error.response?.status === 401) {
        // Token inválido ou expirado
        setAuthToken(null);
        setCurrentScreen('login');
      }
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  };

  const handleLogin = async () => {
    setIsLoading(true);
    setLoginError(null);
    
    try {
      const formData = new FormData();
      formData.append('username', username);
      formData.append('password', password);
      
      const response = await axios.post(`${API_BASE_URL}/token`, formData);
      
      if (response.data.access_token) {
        setAuthToken(response.data.access_token);
        setCurrentScreen('dashboard');
      }
    } catch (error) {
      console.error('Erro de login:', error);
      setLoginError('Credenciais inválidas. Tente novamente.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRefresh = () => {
    setRefreshing(true);
    fetchData();
  };

  const executeOrder = async (side: 'BUY' | 'SELL') => {
    setOrderStatus('pending');
    setOrderMessage(`Executando ordem: ${side} ${orderAmount} ${symbol}`);
    
    try {
      const response = await apiClient.post('/api/trade', {
        symbol: symbol,
        side: side,
        quantity: parseFloat(orderAmount),
        type: 'MARKET'
      });
      
      setOrderStatus('success');
      setOrderMessage(`Ordem executada com sucesso: ${response.data.order.executedQty} @ ${response.data.order.price}`);
      
      // Atualizar dados
      fetchData();
    } catch (error) {
      console.error('Erro ao executar ordem:', error);
      setOrderStatus('error');
      if (axios.isAxiosError(error)) {
        setOrderMessage(`Erro: ${error.response?.data?.detail || error.message}`);
      } else {
        setOrderMessage('Erro ao executar a ordem');
      }
    }
  };

  const handleLogout = () => {
    setAuthToken(null);
    setCurrentScreen('login');
    setAccountInfo(null);
    setPredictions(null);
    setMarketData([]);
  };

  // Tela de Login
  if (currentScreen === 'login') {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" />
        <View style={styles.loginForm}>
          <Text style={styles.title}>Sistema de Trading</Text>
          
          {loginError && (
            <View style={styles.errorContainer}>
              <Text style={styles.errorText}>{loginError}</Text>
            </View>
          )}
          
          <View style={styles.inputGroup}>
            <Text style={styles.label}>Usuário</Text>
            <TextInput
              style={styles.input}
              value={username}
              onChangeText={setUsername}
              autoCapitalize="none"
            />
          </View>
          
          <View style={styles.inputGroup}>
            <Text style={styles.label}>Senha</Text>
            <TextInput
              style={styles.input}
              value={password}
              onChangeText={setPassword}
              secureTextEntry
            />
          </View>
          
          <TouchableOpacity
            style={styles.button}
            onPress={handleLogin}
            disabled={isLoading}
          >
            {isLoading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.buttonText}>Entrar</Text>
            )}
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  // Tela principal
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      <ScrollView
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={handleRefresh} />
        }
      >
        <View style={styles.header}>
          <Text style={styles.title}>Dashboard de Trading</Text>
          
          <View style={styles.symbolSelector}>
            <TouchableOpacity
              style={[
                styles.symbolButton,
                symbol === 'BTCUSDT' && styles.symbolButtonActive
              ]}
              onPress={() => setSymbol('BTCUSDT')}
            >
              <Text style={symbol === 'BTCUSDT' ? styles.symbolTextActive : styles.symbolText}>
                BTC
              </Text>
            </TouchableOpacity>
            
            <TouchableOpacity
              style={[
                styles.symbolButton,
                symbol === 'ETHUSDT' && styles.symbolButtonActive
              ]}
              onPress={() => setSymbol('ETHUSDT')}
            >
              <Text style={symbol === 'ETHUSDT' ? styles.symbolTextActive : styles.symbolText}>
                ETH
              </Text>
            </TouchableOpacity>
            
            <TouchableOpacity
              style={[
                styles.symbolButton,
                symbol === 'ADAUSDT' && styles.symbolButtonActive
              ]}
              onPress={() => setSymbol('ADAUSDT')}
            >
              <Text style={symbol === 'ADAUSDT' ? styles.symbolTextActive : styles.symbolText}>
                ADA
              </Text>
            </TouchableOpacity>
          </View>
        </View>
        
        {/* Balances */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Saldo da Conta</Text>
          {accountInfo ? (
            <View style={styles.balancesContainer}>
              {accountInfo.balances
                .filter(b => ['BTC', 'ETH', 'USDT'].includes(b.asset) && parseFloat(b.free) > 0)
                .map((balance) => (
                  <View key={balance.asset} style={styles.balanceItem}>
                    <Text style={styles.assetName}>{balance.asset}</Text>
                    <Text style={styles.assetBalance}>{parseFloat(balance.free).toFixed(
                      balance.asset === 'USDT' ? 2 : 6
                    )}</Text>
                  </View>
                ))}
            </View>
          ) : (
            <ActivityIndicator />
          )}
        </View>
        
        {/* Preço e Previsão */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Preço Atual - {symbol}</Text>
          
          {predictions ? (
            <View>
              <Text style={styles.priceText}>
                ${predictions.current_price.toFixed(2)}
              </Text>
              
              <View style={[
                styles.signalContainer,
                predictions.signal === 'comprar' ? styles.buySignal :
                predictions.signal === 'vender' ? styles.sellSignal :
                styles.holdSignal
              ]}>
                <Text style={styles.signalText}>
                  {predictions.signal.toUpperCase()} 
                  {' '}({(predictions.percent_change * 100).toFixed(2)}%)
                </Text>
              </View>
            </View>
          ) : (
            <ActivityIndicator />
          )}
        </View>
        
        {/* Gráfico */}
        {marketData.length > 0 && (
          <View style={styles.chartContainer}>
            <Text style={styles.sectionTitle}>Histórico de Preços</Text>
            <LineChart
              data={{
                labels: ['24h', '18h', '12h', '6h', 'Agora'],
                datasets: [
                  {
                    data: [
                      marketData[0].close,
                      marketData[6].close,
                      marketData[12].close,
                      marketData[18].close,
                      marketData[marketData.length - 1].close
                    ]
                  }
                ]
              }}
              width={Dimensions.get('window').width - 16}
              height={220}
              chartConfig={{
                backgroundColor: '#ffffff',
                backgroundGradientFrom: '#ffffff',
                backgroundGradientTo: '#f0f0f0',
                decimalPlaces: 2,
                color: (opacity = 1) => `rgba(0, 122, 255, ${opacity})`,
                labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
                style: {
                  borderRadius: 16
                },
                propsForDots: {
                  r: '6',
                  strokeWidth: '2',
                  stroke: '#0070f3'
                }
              }}
              bezier
              style={{
                marginVertical: 8,
                borderRadius: 16
              }}
            />
          </View>
        )}
        
        {/* Painel de Ordens */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Executar Ordem</Text>
          
          <View style={styles.orderPanel}>
            <View style={styles.inputGroup}>
              <Text style={styles.label}>Quantidade</Text>
              <TextInput
                style={styles.input}
                value={orderAmount}
                onChangeText={setOrderAmount}
                keyboardType="numeric"
              />
            </View>
            
            <View style={styles.orderButtons}>
              <TouchableOpacity
                style={[styles.orderButton, styles.buyButton]}
                onPress={() => executeOrder('BUY')}
              >
                <Text style={styles.orderButtonText}>COMPRAR</Text>
              </TouchableOpacity>
              
              <TouchableOpacity
                style={[styles.orderButton, styles.sellButton]}
                onPress={() => executeOrder('SELL')}
              >
                <Text style={styles.orderButtonText}>VENDER</Text>
              </TouchableOpacity>
            </View>
          </View>
          
          {orderMessage && (
            <View style={[
              styles.orderMessage,
              orderStatus === 'success' ? styles.successMessage :
              orderStatus === 'error' ? styles.errorMessage :
              styles.pendingMessage
            ]}>
              <Text style={styles.orderMessageText}>{orderMessage}</Text>
            </View>
          )}
        </View>
        
        {/* Botão de Logout */}
        <View style={styles.logoutContainer}>
          <TouchableOpacity
            style={styles.logoutButton}
            onPress={handleLogout}
          >
            <Text style={styles.logoutButtonText}>Sair</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9f9f9',
  },
  loginForm: {
    flex: 1,
    justifyContent: 'center',
    padding: 16,
  },
  header: {
    padding: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  inputGroup: {
    marginBottom: 16,
  },
  label: {
    fontSize: 16,
    marginBottom: 8,
    color: '#333',
  },
  input: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
  },
  button: {
    backgroundColor: '#0070f3',
    borderRadius: 8,
    padding: 16,
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  errorContainer: {
    backgroundColor: '#ffebee',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  errorText: {
    color: '#c62828',
  },
  section: {
    backgroundColor: '#fff',
    padding: 16,
    margin: 8,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
    elevation: 2,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 12,
    color: '#333',
  },
  symbolSelector: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 12,
  },
  symbolButton: {
    flex: 1,
    padding: 12,
    alignItems: 'center',
    borderRadius: 8,
    backgroundColor: '#f0f0f0',
    marginHorizontal: 4,
  },
  symbolButtonActive: {
    backgroundColor: '#0070f3',
  },
  symbolText: {
    fontWeight: 'bold',
    color: '#333',
  },
  symbolTextActive: {
    fontWeight: 'bold',
    color: '#fff',
  },
  balancesContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
  },
  balanceItem: {
    width: '30%',
    padding: 8,
    backgroundColor: '#f0f7ff',
    borderRadius: 8,
    marginBottom: 8,
  },
  assetName: {
    fontWeight: 'bold',
    fontSize: 16,
    color: '#333',
  },
  assetBalance: {
    fontSize: 16,
    color: '#0070f3',
    marginTop: 4,
  },
  priceText: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'center',
    marginVertical: 8,
  },
  signalContainer: {
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 8,
  },
  buySignal: {
    backgroundColor: '#e8f5e9',
  },
  sellSignal: {
    backgroundColor: '#ffebee',
  },
  holdSignal: {
    backgroundColor: '#f5f5f5',
  },
  signalText: {
    fontWeight: 'bold',
    fontSize: 16,
  },
  chartContainer: {
    backgroundColor: '#fff',
    padding: 16,
    margin: 8,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
    elevation: 2,
  },
  orderPanel: {
    marginTop: 8,
  },
  orderButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  orderButton: {
    flex: 1,
    padding: 16,
    alignItems: 'center',
    borderRadius: 8,
    marginHorizontal: 4,
  },
  buyButton: {
    backgroundColor: '#4caf50',
  },
  sellButton: {
    backgroundColor: '#f44336',
  },
  orderButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
  },
  orderMessage: {
    padding: 12,
    borderRadius: 8,
    marginTop: 16,
  },
  successMessage: {
    backgroundColor: '#e8f5e9',
  },
  errorMessage: {
    backgroundColor: '#ffebee',
  },
  pendingMessage: {
    backgroundColor: '#fff8e1',
  },
  orderMessageText: {
    fontSize: 14,
  },
  logoutContainer: {
    margin: 16,
    alignItems: 'center',
  },
  logoutButton: {
    backgroundColor: '#f0f0f0',
    padding: 12,
    borderRadius: 8,
    width: '50%',
    alignItems: 'center',
  },
  logoutButtonText: {
    color: '#757575',
    fontWeight: 'bold',
  }
});

export default App;
