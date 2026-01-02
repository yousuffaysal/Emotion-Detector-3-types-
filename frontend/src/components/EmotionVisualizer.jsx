import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const EmotionVisualizer = ({ emotion }) => {
    // Config for each emotion (Subtle background washes)
    const config = {
        Happy: {
            bg: 'linear-gradient(120deg, #fffff0 0%, #ffe4d6 100%)', // Very subtle peach
            accent: '#e0501a'
        },
        Sad: {
            bg: 'linear-gradient(120deg, #fffff0 0%, #e6eef5 100%)', // Very subtle blue grey
            accent: '#4a6fa5'
        },
        Neutral: {
            bg: '#fffff0',
            accent: '#1a1a1a'
        }
    };

    const currentConfig = config[emotion] || config.Neutral;

    return (
        <motion.div
            className="absolute inset-0 w-full h-full -z-10 transition-colors duration-1000 ease-in-out"
            animate={{ background: currentConfig.bg }}
            initial={false}
        />
    );
};

export default EmotionVisualizer;
