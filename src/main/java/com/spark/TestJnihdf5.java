package com.spark;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;


public class TestJnihdf5 {

    private static Connection dbConnection = null;
    public static final String DB_URL = "jdbc:postgresql://localhost:5432/postgres";
    public static final String DB_USER = "postgres";
    public static final String DB_PASSWORD = null;

    public static Connection getDbConnection() throws Exception {
        if (dbConnection == null || dbConnection.isClosed()) {
            dbConnection = DriverManager.getConnection(DB_URL, DB_USER, DB_PASSWORD);
            dbConnection.setAutoCommit(true);
        }
        return dbConnection;
    }

    public static void main(String[] args) throws Exception {

        Connection conn = getDbConnection();

        int id = 5001;
        float riskPropensity = 0.342F;
        float financialStatus = 0.342F;
        int income = 1;
        int accumulation = 1;
        System.out.println("queryDb" + ", id: " + id +  "riskprop: " + riskPropensity);

        String updateQuery = "UPDATE needs " +
                "SET risk_propensity = " + riskPropensity + ", " +
                "financial_status = " + financialStatus +
                " WHERE id = " + id;

        Statement st = conn.createStatement();
        int numRes = st.executeUpdate(updateQuery);
        System.out.println("numRes: " + numRes);
        System.out.printf("Updated for id = %d risk_propensity = %f and financial_status = %f%n",
                id, riskPropensity, financialStatus);

        String selectQuery = "SELECT * FROM products " +
                "WHERE (" +
                "Income = " + income + " " +
                "OR Accumulation = " + accumulation + ") " +
                "AND Risk <= " + riskPropensity;

        ResultSet rs = st.executeQuery(selectQuery);
        int count = 0;
        while (rs.next()) {
            count++;
        }
        System.out.printf("Advised %d products for id = %d%n", count, id);
        st.close();
        rs.close();
    }
}