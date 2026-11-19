# Piloto de validación y KPIs

## Piloto: venta cruzada Cocinas → Baños en Antioquia

**Por qué este piloto.** Junta los tres hallazgos más sólidos del análisis:

- **Antioquia · Gen X** es la oportunidad de activación número 1 del índice (`08`). Es el departamento con más clientes potenciales (≈16.000) y además tiene 11 tiendas.
- **Los clientes consolidados están sobrerrepresentados en Antioquia**, y en ese segmento la regla Cocinas → Baños es la más fuerte. Tiene lift 2,8 y 43 % de confianza dentro del mismo pedido (`06`).
- **Los consolidados son los que más recompran:** 40 % en la validación fuera de tiempo (`10`).

**Hipótesis.** Quien compra Cocinas y no lleva Baños en el mismo pedido es un cliente en plena remodelación. Si se le hace una oferta de Baños oportuna, compra más de lo que compraría solo.

**Línea base (datos reales).** Hoy, de los consumidores que compran Cocinas sin llevar Baños, solo el **6,0 %** compra Baños en los 60 días siguientes en Antioquia (5,1 % a nivel nacional). Cuando compran, gastan en Baños un promedio de ≈245.000 en unidades de la base.

| Elemento | Definición |
|---|---|
| Población | Consumidores de Antioquia que compraron Cocinas en los últimos 9 meses sin comprar Baños **y que autorizaron el tratamiento de datos** (`Tratamiento Datos = SÍ`). Corona confirmó que no se puede contactar a quien no autorizó. |
| Tamaño | ≈830 clientes: ≈680 acumulados más ≈150 nuevos durante el piloto. Antioquia genera ≈76 elegibles con consentimiento al mes. |
| Diseño | Asignación aleatoria 50/50: **tratamiento** (≈415) frente a **control** (≈415, sin contacto). |
| Intervención | Oferta de Baños por el canal del asesor y por WhatsApp o correo, con un incentivo de instalación (el negocio Servicios, que también tiene afinidad alta con Baños). |
| Duración | 8 semanas de contacto y 60 días de medición por cliente. |
| Poder estadístico | Con ≈415 por grupo se detecta un aumento de **+5 puntos** (de 6 % a 11 %) con 95 % de confianza y 80 % de poder. |

**KPIs del piloto**

- **Principal:** tasa de compra de Baños a 60 días, en tratamiento frente a control.
- **Secundarios:**
  - valor incremental por cliente contactado;
  - ticket del pedido de Baños;
  - % de clientes que pasan a un segmento de mayor vínculo.
- **De control (*guardrails*):**
  - tasa de bajas o quejas por contacto;
  - que la venta de Baños no caiga en el grupo control. Si cae, hubo canibalización entre tiendas.

**Regla de decisión.** Escalar a Bogotá y Valle, que tienen el mismo patrón y más volumen, si se cumplen las tres condiciones:

1. El aumento es de al menos 3 puntos.
2. El intervalo de confianza no incluye 0.
3. El valor incremental cubre el costo del incentivo.

## KPIs para medir el impacto del modelo en el negocio

| KPI | Fórmula | Frecuencia | De dónde sale |
|---|---|---|---|
| Penetración por generación y departamento | Clientes ÷ población DANE × 1.000 | Trimestral | `07` |
| Cierre de brecha en departamentos priorizados | Clientes nuevos en el top 10 del índice ÷ clientes potenciales | Trimestral | `08` |
| Recompra por segmento | % de clientes con un pedido nuevo en 12 meses | Trimestral | `10` |
| Migración entre segmentos | % de "recientes" que pasan a "en desarrollo" o "consolidados" | Semestral | `05` |
| Venta cruzada | % de pedidos de Cocinas, Pinturas o Servicios que incluyen Baños | Mensual | `06` |
| Captación Gen Z | % de clientes nuevos que son Gen Z (hoy 8,3 %) | Mensual | `09` |
