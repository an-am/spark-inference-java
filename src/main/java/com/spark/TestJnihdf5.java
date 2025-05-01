package com.spark;

public class TestJnihdf5 {
    static {
        System.load("/usr/local/lib/jnihdf5/libjnihdf5.dylib");
    }
    public static void main(String[] args) {
        System.out.println("Loaded jnihdf5 successfully!");
    }
}