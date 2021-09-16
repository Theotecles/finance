SELECT symbol,
	   data_date,
	   adjusted_close
FROM finance.dbo.s_and_p_daily
WHERE symbol = 'DIS'
AND data_date >= '2020-12-01'
ORDER BY data_date DESC
