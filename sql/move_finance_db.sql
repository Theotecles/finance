ALTER DATABASE finance   
    MODIFY FILE ( NAME = finance,   
                  FILENAME = 'E:\SQLServer\Data\finance_Data.mdf');  
GO
 
ALTER DATABASE finance   
    MODIFY FILE ( NAME = finance_Log,   
                  FILENAME = 'E:\SQLServer\Logs\finance_Log.ldf');  
GO