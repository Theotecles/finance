SELECT symbol,
	   data_date,
	   adjusted_close
FROM finance.dbo.s_and_p_daily
WHERE symbol = 'DIS'
ORDER BY data_date DESC