package visionai.database;

import org.apache.commons.dbcp2.BasicDataSource;
import java.sql.Connection;
import java.sql.SQLException;

public class DBConnection {
    private static final BasicDataSource dataSource = new BasicDataSource();
    
    static {
        dataSource.setUrl("jdbc:postgresql://localhost:5432/visionai");
        dataSource.setUsername("your_username");
        dataSource.setPassword("your_password");
        dataSource.setMinIdle(5);
        dataSource.setMaxIdle(10);
        dataSource.setMaxOpenPreparedStatements(100);
    }
    
    public static Connection getConnection() throws SQLException {
        return dataSource.getConnection();
    }
    
    public static void closeDataSource() throws SQLException {
        dataSource.close();
    }
}