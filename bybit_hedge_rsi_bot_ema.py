import config
import time
from pybit.unified_trading import HTTP
from decimal import Decimal, ROUND_FLOOR
from datetime import datetime
import pandas as pd
import numpy as np

# Inicializar sesión con Bybit
session = HTTP(
    testnet=False,
    api_key=config.api_key,
    api_secret=config.api_secret
)

def log_mensaje(mensaje):
    """
    Registra un mensaje con marca de tiempo UTC
    """
    tiempo_utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{tiempo_utc} UTC] {mensaje}")

def calcular_rsi_wilder(symbol, intervalo="1", periodo=14):
    """
    Calcula el RSI usando el método de Wilder con EMA
    """
    try:
        kline = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=intervalo,
            limit=500
        )
        
        if 'result' not in kline or 'list' not in kline['result']:
            log_mensaje("Error: No se pudieron obtener datos de velas")
            return None

        # Convertir datos a DataFrame
        df = pd.DataFrame([
            {
                'timestamp': float(candle[0]),
                'close': float(candle[4]),
            }
            for candle in kline['result']['list']
        ])

        # Ordenar por timestamp
        df = df.sort_values('timestamp')
        
        # Calcular cambios
        df['cambio'] = df['close'].diff()
        
        # Separar ganancias y pérdidas
        df['ganancia'] = df['cambio'].apply(lambda x: x if x > 0 else 0)
        df['perdida'] = df['cambio'].apply(lambda x: abs(x) if x < 0 else 0)
        
        # Calcular EMA de ganancias y pérdidas
        alpha = 1/periodo
        df['avg_gain'] = df['ganancia'].ewm(alpha=alpha, min_periods=periodo).mean()
        df['avg_loss'] = df['perdida'].ewm(alpha=alpha, min_periods=periodo).mean()

        # Calcular RS y RSI
        df['rs'] = df['avg_gain'] / df['avg_loss']
        df['rsi'] = 100 - (100 / (1 + df['rs']))

        rsi_actual = round(df['rsi'].iloc[-1], 2)
        log_mensaje(f"RSI calculado: {rsi_actual}")
        
        # Mostrar últimos valores para verificación
        ultimos_valores = df.tail(5)
        log_mensaje("\nÚltimos 5 valores:")
        for idx, row in ultimos_valores.iterrows():
            log_mensaje(f"Precio: {row['close']}, RSI: {round(row['rsi'], 2)}")

        return rsi_actual

    except Exception as e:
        log_mensaje(f"Error en el cálculo del RSI: {e}")
        return None

def verificar_posicion_abierta(symbol, side):
    """
    Verifica si hay una posición abierta
    """
    try:
        positions = session.get_positions(
            category="linear",
            symbol=symbol
        )
        for position in positions['result']['list']:
            if float(position['size']) > 0 and position['side'] == side:
                return True
        return False
    except Exception as e:
        log_mensaje(f"Error al verificar posición: {e}")
        return True

def actualizar_stop_loss(symbol, side, precio_actual, stop_loss_inicial, trailing_step):
    """
    Actualiza el stop loss con trailing
    """
    try:
        positions = session.get_positions(
            category="linear",
            symbol=symbol
        )
        
        for position in positions['result']['list']:
            if float(position['size']) > 0 and position['side'] == side:
                entrada_precio = float(position['avgPrice'])
                stop_loss_actual = float(position['stopLoss']) if position['stopLoss'] != '0' else None
                
                if side == "Buy":
                    profit_actual = (precio_actual - entrada_precio) / entrada_precio * 100
                    if profit_actual >= trailing_step:
                        steps = int(profit_actual / trailing_step)
                        new_stop_loss = entrada_precio * (1 + (steps - 1) * trailing_step / 100)
                        new_stop_loss = max(new_stop_loss, stop_loss_actual if stop_loss_actual else entrada_precio * (1 - stop_loss_inicial / 100))
                        
                        if stop_loss_actual is None or new_stop_loss > stop_loss_actual:
                            session.set_trading_stop(
                                category="linear",
                                symbol=symbol,
                                stopLoss=new_stop_loss,
                                positionIdx=1
                            )
                            log_mensaje(f"Stop Loss actualizado (Long): {new_stop_loss}")
                else:
                    profit_actual = (entrada_precio - precio_actual) / entrada_precio * 100
                    if profit_actual >= trailing_step:
                        steps = int(profit_actual / trailing_step)
                        new_stop_loss = entrada_precio * (1 - (steps - 1) * trailing_step / 100)
                        new_stop_loss = min(new_stop_loss, stop_loss_actual if stop_loss_actual else entrada_precio * (1 + stop_loss_inicial / 100))
                        
                        if stop_loss_actual is None or new_stop_loss < stop_loss_actual:
                            session.set_trading_stop(
                                category="linear",
                                symbol=symbol,
                                stopLoss=new_stop_loss,
                                positionIdx=2
                            )
                            log_mensaje(f"Stop Loss actualizado (Short): {new_stop_loss}")
                            
    except Exception as e:
        log_mensaje(f"Error al actualizar stop loss: {e}")

def obtener_precio_actual(symbol):
    """
    Obtiene el precio actual del mercado
    """
    try:
        ticker = session.get_tickers(
            category="linear",
            symbol=symbol
        )
        return float(ticker['result']['list'][0]['lastPrice'])
    except Exception as e:
        log_mensaje(f"Error al obtener precio: {e}")
        return None

def colocar_orden(symbol, side, capital_usdt, precio_entrada, stop_loss_pct, take_profit_pct):
    """
    Coloca una orden con gestión de riesgos
    """
    try:
        # Calcular cantidad
        qty = capital_usdt / precio_entrada
        
        # Calcular niveles
        if side == "Buy":
            stop_loss = precio_entrada * (1 - stop_loss_pct / 100)
            take_profit = precio_entrada * (1 + take_profit_pct / 100)
        else:
            stop_loss = precio_entrada * (1 + stop_loss_pct / 100)
            take_profit = precio_entrada * (1 - take_profit_pct / 100)

        # Colocar orden
        orden = session.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=qty,
            positionIdx=1 if side == "Buy" else 2
        )
        
        if orden['retCode'] == 0:
            # Configurar SL/TP
            session.set_trading_stop(
                category="linear",
                symbol=symbol,
                stopLoss=stop_loss,
                takeProfit=take_profit,
                positionIdx=1 if side == "Buy" else 2
            )
            
            log_mensaje(f"Orden {side} colocada: Cantidad={qty}, SL={stop_loss}, TP={take_profit}")
            return True
            
        return False
        
    except Exception as e:
        log_mensaje(f"Error al colocar orden: {e}")
        return False

def ejecutar_bot():
    log_mensaje("=== BOT DE TRADING CON RSI WILDER Y STOP LOSS AUTOMÁTICO ===")
    log_mensaje(f"Usuario: REN674")
    log_mensaje(f"Inicio: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Configuración
    symbol = input("Símbolo (ej: BTC): ").upper() + "USDT"
    capital_usdt = float(input("Capital USDT por operación: "))
    stop_loss_pct = float(input("Stop Loss inicial %: "))
    take_profit_pct = float(input("Take Profit %: "))
    trailing_step = float(input("Paso del trailing stop %: "))
    rsi_long = float(input("RSI para Long (ej: 30): "))
    rsi_short = float(input("RSI para Short (ej: 70): "))
    intervalo = input("Temporalidad (1, 3, 5, 15, 30, 60, etc): ")

    posicion_actual = None

    while True:
        try:
            precio_actual = obtener_precio_actual(symbol)
            if precio_actual is None:
                continue

            # Si hay posición abierta
            if posicion_actual:
                actualizar_stop_loss(symbol, posicion_actual, precio_actual, stop_loss_pct, trailing_step)
                
                if not verificar_posicion_abierta(symbol, posicion_actual):
                    log_mensaje(f"Posición {posicion_actual} cerrada")
                    posicion_actual = None
                    time.sleep(60)
                    continue
                
                time.sleep(10)
                continue

            # Calcular RSI y buscar entradas
            rsi = calcular_rsi_wilder(symbol, intervalo)

            if rsi is not None:
                log_mensaje(f"RSI actual: {rsi}")

                if rsi < rsi_long and not verificar_posicion_abierta(symbol, "Buy"):
                    log_mensaje("Señal de compra detectada")
                    if colocar_orden(symbol, "Buy", capital_usdt, precio_actual, stop_loss_pct, take_profit_pct):
                        posicion_actual = "Buy"

                elif rsi > rsi_short and not verificar_posicion_abierta(symbol, "Sell"):
                    log_mensaje("Señal de venta detectada")
                    if colocar_orden(symbol, "Sell", capital_usdt, precio_actual, stop_loss_pct, take_profit_pct):
                        posicion_actual = "Sell"

            time.sleep(60)

        except Exception as e:
            log_mensaje(f"Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    ejecutar_bot()